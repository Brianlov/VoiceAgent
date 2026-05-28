import os
import asyncio
import re
from neo4j import AsyncGraphDatabase
import logging
from typing import List
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from pipecat.frames.frames import (
    TextFrame, Frame, InterimTranscriptionFrame, 
    UserStoppedSpeakingFrame, StartFrame, TranscriptionFrame, 
    LLMMessagesAppendFrame, LLMMessagesFrame, LLMContextFrame
)
from pipecat.processors.frame_processor import FrameProcessor
import pipecat.frames.frames

logger = logging.getLogger(__name__)

RELATION_WEIGHTS = {
    "WRITTEN_BY": 1.0,
    "DIRECTED_BY": 1.0,
    "ACTED_IN": 0.8,
    "PART_OF": 0.7,
    "PERFORMER": 0.3,
    "MENTIONED_IN": 0.1,
}

class GraphRAGProcessorV3(FrameProcessor):
    LAST_QUERY = ""
    LAST_CONTEXT = ""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, neo4j_db: str, knowledge_base: List[str] = None):
        super().__init__()
        
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_db = neo4j_db
        self.driver = AsyncGraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        print("🔧 GraphRAG [PURE v3 - Path Scoring]: Loading Embedding Model (all-mpnet-base-v2)...")
        self.model = SentenceTransformer("all-mpnet-base-v2")
        
        self._initialized = False
        self.saved_stop_frame = None

    async def _initialize_graph(self):
        """Verify Neo4j Connection."""
        if self._initialized: return

        print(f"🧠 GraphRAG [PURE v3]: Verifying Neo4j Connection at {self.neo4j_uri}...")
        try:
            await self.driver.verify_connectivity()
            print(f"✅ GraphRAG [PURE v3]: Connected to Neo4j successfully.")
            self._initialized = True
        except Exception as e:
            print(f"❌ GraphRAG [PURE v3]: Failed to connect to Neo4j: {e}")

    async def retrieve(self, query: str, as_list: bool = False):
        print(f"🚨 RETRIEVE [PURE GraphRAG v3] CALLED with: '{query}'")
        if not self._initialized:
            await self._initialize_graph()
            
        if not self._initialized:
             return [] if as_list else ""

        # 1. Generate Embedding for the User Query
        query_vector = await asyncio.to_thread(self.model.encode, query)
        query_vector_list = query_vector.tolist()

        context_parts = []
        raw_chunks = []
        
        try:
            async with self.driver.session(database=self.neo4j_db) as session:
                # ---------------------------------------------------------
                # PHASE 1: Retrieve Raw Paths (1 to 3 hops) from Neo4j
                # ---------------------------------------------------------
                print(f"🔍 GraphRAG [PURE v3]: Executing path retrieval Cypher...")
                path_cypher = """
                CALL db.index.vector.queryNodes('entity_vector_index', 5, $query_vector) 
                YIELD node AS e, score AS entry_score
                MATCH p=(e)-[r*1..3]-(n)
                WHERE ALL(rel IN r WHERE type(rel) <> 'MENTIONED_IN')
                RETURN p, entry_score
                LIMIT 300
                """
                result = await session.run(path_cypher, query_vector=query_vector_list)
                records = [record async for record in result]
                print(f"📦 GraphRAG [PURE v3]: Retrieved {len(records)} raw paths")

                if not records:
                    print(f"❌ GraphRAG [PURE v3]: No paths found.")
                    return [] if as_list else ""

                # ---------------------------------------------------------
                # PHASE 2: Extract Path Features in Python
                # ---------------------------------------------------------
                parsed_paths = []
                for record in records:
                    p = record['p']
                    entry_score = record.get('entry_score', 0)
                    
                    nodes = list(p.nodes)
                    rels = list(p.relationships)
                    
                    # Construct textual representation of the path
                    path_tokens = []
                    for i in range(len(nodes)):
                        node_name = nodes[i].get('name', 'Unknown')
                        path_tokens.append(str(node_name))
                        if i < len(rels):
                            path_tokens.append(str(rels[i].type))
                            
                    path_text = " ".join(path_tokens)
                    
                    # Calculate relation score based on priors
                    relation_score = sum(RELATION_WEIGHTS.get(r.type, 0.5) for r in rels)
                    
                    parsed_paths.append({
                        'nodes': nodes,
                        'rels': rels,
                        'path_text': path_text,
                        'entry_score': entry_score,
                        'relation_score': relation_score,
                        'depth_penalty': len(rels) * 0.2
                    })
                
                # ---------------------------------------------------------
                # PHASE 3: Path Semantic Embedding & Scoring
                # ---------------------------------------------------------
                path_texts = [p['path_text'] for p in parsed_paths]
                path_embeddings = await asyncio.to_thread(self.model.encode, path_texts)
                
                norm_q = np.linalg.norm(query_vector)
                for i, p_data in enumerate(parsed_paths):
                    p_emb = path_embeddings[i]
                    norm_p = np.linalg.norm(p_emb)
                    sim_score = 0.0
                    if norm_q > 0 and norm_p > 0:
                        sim_score = np.dot(query_vector, p_emb) / (norm_q * norm_p)
                    
                    p_data['semantic_score'] = sim_score
                    # Combine semantic similarity, relation prior (weighted down), and depth penalty
                    p_data['final_score'] = sim_score + (p_data['relation_score'] * 0.1) - p_data['depth_penalty']

                # ---------------------------------------------------------
                # PHASE 4: Beam Pruning (Keep Top-10 Paths)
                # ---------------------------------------------------------
                parsed_paths.sort(key=lambda x: x['final_score'], reverse=True)
                top_paths = parsed_paths[:10]
                
                print(f"🎯 GraphRAG [PURE v3]: Top path text: '{top_paths[0]['path_text']}' (Score: {top_paths[0]['final_score']:.3f})")

                # Extract unique entities from the winning paths
                top_entity_names = set()
                for p_data in top_paths:
                    for n in p_data['nodes']:
                        name = n.get('name')
                        if name:
                            top_entity_names.add(name)

                # ---------------------------------------------------------
                # PHASE 5: Retrieve Chunks ONLY from Winning Entities
                # ---------------------------------------------------------
                if top_entity_names:
                    chunk_cypher = """
                    UNWIND $entity_names AS e_name
                    MATCH (e:Entity {name: e_name})-[:MENTIONED_IN]->(c:Context)
                    RETURN DISTINCT c.text AS chunk_text, c.embedding AS chunk_embedding
                    """
                    chunk_result = await session.run(chunk_cypher, entity_names=list(top_entity_names))
                    chunk_records = await chunk_result.data()
                    
                    candidate_chunks = []
                    seen_texts = set()
                    
                    for cr in chunk_records:
                        chunk_text = cr.get('chunk_text')
                        if chunk_text and chunk_text not in seen_texts:
                            seen_texts.add(chunk_text)
                            
                            chunk_emb = cr.get('chunk_embedding')
                            sim_score = 0.0
                            if chunk_emb and query_vector_list:
                                dot_prod = np.dot(query_vector_list, chunk_emb)
                                norm_c = np.linalg.norm(chunk_emb)
                                if norm_q > 0 and norm_c > 0:
                                    sim_score = dot_prod / (norm_q * norm_c)
                                    
                            candidate_chunks.append({
                                'text': chunk_text,
                                'similarity': sim_score
                            })
                            
                    candidate_chunks.sort(key=lambda x: x['similarity'], reverse=True)
                    top_chunks = [c['text'] for c in candidate_chunks[:5]]
                    
                    if top_chunks:
                        context_parts.append("[GROUNDING CONTEXT]")
                        context_parts.extend(top_chunks)
                        context_parts.append("")
                        raw_chunks.extend(top_chunks)
                        
                    # Add the winning paths so the LLM sees the multi-hop reasoning sequence
                    winning_path_texts = [p['path_text'] for p in top_paths]
                    if winning_path_texts:
                        context_parts.append("[REASONING PATHS]")
                        context_parts.extend(winning_path_texts)
                        context_parts.append("")
                        raw_chunks.extend(winning_path_texts)

        except Exception as e:
            print(f"⚠️ GraphRAG [PURE v3] Retrieval error: {e}")
            
        if not context_parts:
            print(f"❌ GraphRAG [PURE v3]: No context found for query='{query}'")
            return [] if as_list else ""
            
        if as_list:
            return raw_chunks[:15]
            
        final_context = "\n".join(context_parts)
        print(f"✅ GraphRAG [PURE v3]: Retrieved Context Content:\n{'-'*40}\n{final_context}\n{'-'*40}")
        return final_context[:8000]

    async def _flush_stop_frame(self, direction):
        if self.saved_stop_frame:
            await self.push_frame(self.saved_stop_frame, direction)
            self.saved_stop_frame = None

    async def _flush_stop_frame_delayed(self, delay, direction):
        await asyncio.sleep(delay)
        if self.saved_stop_frame:
             await self._flush_stop_frame(direction)

    async def process_frame(self, frame: Frame, direction):
        if isinstance(frame, StartFrame):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterimTranscriptionFrame):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return
            
        if isinstance(frame, UserStoppedSpeakingFrame):
            self.saved_stop_frame = frame
            asyncio.create_task(self._flush_stop_frame_delayed(0.5, direction))
            return

        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text
            if user_text.strip():
                context = await self.retrieve(user_text)
                GraphRAGProcessorV3.LAST_QUERY = user_text
                GraphRAGProcessorV3.LAST_CONTEXT = context
                
                enriched_prompt = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{user_text}\n\nInstructions:\n1. Read the [CONTEXT] carefully. It contains paragraphs and reasoning paths.\n2. Find the exact answer to the [QUESTION] using the reasoning paths if necessary.\n3. Provide a natural, conversational sentence as your answer.\n4. If the answer is not in the [CONTEXT], say: 'I'm sorry, I couldn't find that in the database.'\n"
                frame.text = enriched_prompt
            
            await self.push_frame(frame, direction)
            await self._flush_stop_frame(direction)
            from pipecat.frames.frames import LLMRunFrame
            await self.push_frame(LLMRunFrame(), direction)
            return

        if isinstance(frame, LLMContextFrame):
            messages = frame.context.messages
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            
            if user_messages:
                latest_query = user_messages[-1].get("content", "")
                if isinstance(latest_query, str) and latest_query.strip():
                    context = await self.retrieve(latest_query)
                    GraphRAGProcessorV3.LAST_QUERY = latest_query
                    GraphRAGProcessorV3.LAST_CONTEXT = context
                    
                    enriched_prompt = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{latest_query}\n\nInstructions:\n1. Read the [CONTEXT] carefully. It contains paragraphs and reasoning paths.\n2. Find the exact answer to the [QUESTION] using the reasoning paths if necessary.\n3. Provide a natural, conversational sentence as your answer.\n4. If the answer is not in the [CONTEXT], say: 'I'm sorry, I couldn't find that in the database.'\n"
                    user_messages[-1]["content"] = enriched_prompt
            
            await self.push_frame(frame, direction)
            await self._flush_stop_frame(direction)
            return

        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

class AnswerLoggerProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.current_answer = ""

    async def process_frame(self, frame, direction):
        frame_type = type(frame).__name__
        if "TextFrame" in frame_type:
            self.current_answer += frame.text
        elif frame_type in ["LLMFullResponseEndFrame", "TTSStoppedFrame", "UserStartedSpeakingFrame", "UserStoppedSpeakingFrame"]:
            if self.current_answer.strip():
                with open("graphrag_log.txt", "a", encoding="utf-8") as f:
                    log_entry = (
                        f"--- PURE GraphRAG v3 Session ---\n"
                        f"Retrieved Context:\n{GraphRAGProcessorV3.LAST_CONTEXT}\n\n"
                        f"User Question:\n{GraphRAGProcessorV3.LAST_QUERY}\n\n"
                        f"Final Answer:\n{self.current_answer}\n"
                        f"{'-'*30}\n\n"
                    )
                    f.write(log_entry)
                self.current_answer = ""
                
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
