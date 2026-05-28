import os
import asyncio
import re
from neo4j import AsyncGraphDatabase
import logging
from typing import List
import json

from sentence_transformers import SentenceTransformer
from pipecat.frames.frames import (
    TextFrame, Frame, InterimTranscriptionFrame, 
    UserStoppedSpeakingFrame, StartFrame, TranscriptionFrame, 
    LLMMessagesAppendFrame, LLMMessagesFrame, LLMContextFrame
)
from pipecat.processors.frame_processor import FrameProcessor
import pipecat.frames.frames

logger = logging.getLogger(__name__)

class GraphRAGProcessor(FrameProcessor):
    LAST_QUERY = ""
    LAST_CONTEXT = ""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, neo4j_db: str, knowledge_base: List[str] = None):
        super().__init__()
        
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_db = neo4j_db
        self.driver = AsyncGraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        # Load the same model used for embedding the CSVs
        print("🔧 GraphRAG: Loading Embedding Model (all-mpnet-base-v2)...")
        self.model = SentenceTransformer("all-mpnet-base-v2")
        
        self._initialized = False
        
        # Buffer for reordering frames
        self.saved_stop_frame = None

    async def _initialize_graph(self):
        """Verify Neo4j Connection."""
        if self._initialized: return

        print(f"🧠 GraphRAG: Verifying Neo4j Connection at {self.neo4j_uri}...")
        try:
            await self.driver.verify_connectivity()
            print(f"✅ GraphRAG: Connected to Neo4j successfully.")
            self._initialized = True
        except Exception as e:
            print(f"❌ GraphRAG: Failed to connect to Neo4j: {e}")

    async def retrieve(self, query: str, as_list: bool = False):
        print(f"🚨 RETRIEVE CALLED with: '{query}'")
        if not self._initialized:
            await self._initialize_graph()
            
        if not self._initialized:
             return [] if as_list else ""

        # 1. Generate Embedding for the User Query using the local model
        query_vector = self.model.encode(query).tolist()

        # 2. Hybrid Vector + Graph Search
        # This query performs a true hybrid search by querying BOTH the context_vector_index 
        # (for direct semantic chunk matching) and the entity_vector_index (for multi-hop traversal),
        # then unioning the results to achieve maximum retrieval accuracy.
        cypher = """
        // Part 1: Search Context vector index
        CALL db.index.vector.queryNodes('context_vector_index', 3, $query_vector) 
        YIELD node AS c, score AS context_score
        WITH c, context_score

        // Get entities mentioned in these contexts
        OPTIONAL MATCH (e_from_c:Entity)-[:MENTIONED_IN]->(c)
        WITH c, context_score, collect(DISTINCT e_from_c) AS entities_from_c

        // Get relationships for those entities
        UNWIND (case when size(entities_from_c) > 0 then entities_from_c else [null] end) AS e_c
        OPTIONAL MATCH (e_c)-[r_c]->(neighbor_c:Entity)
        WHERE type(r_c) <> "MENTIONED_IN" AND neighbor_c IN entities_from_c

        WITH c, context_score, 
             collect(DISTINCT coalesce(e_c.name, e_c.id) + " " + type(r_c) + " " + coalesce(neighbor_c.name, neighbor_c.id)) AS c_facts,
             collect(DISTINCT coalesce(e_c.name, e_c.id) + ": " + coalesce(e_c.description, "")) AS c_descs
             
        RETURN 
          c.text AS chunk_text,
          context_score AS final_score,
          c_facts AS all_facts,
          c_descs AS all_descriptions
        ORDER BY final_score DESC
        LIMIT 3

        UNION

        // Part 2: Search Entity vector index
        CALL db.index.vector.queryNodes('entity_vector_index', 3, $query_vector) 
        YIELD node AS e, score AS entity_score

        // Get relationships from these entities
        OPTIONAL MATCH (e)-[r]->(neighbor:Entity)
        WHERE type(r) <> "MENTIONED_IN"
        WITH e, entity_score, collect(DISTINCT coalesce(e.name, e.id) + " " + type(r) + " " + coalesce(neighbor.name, neighbor.id)) AS e_facts

        // Get contexts directly linked to these entities
        OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c_direct:Context)
        WITH e, entity_score, e_facts, c_direct
        WHERE c_direct IS NOT NULL

        RETURN 
          c_direct.text AS chunk_text,
          entity_score AS final_score,
          e_facts AS all_facts,
          collect(DISTINCT coalesce(e.name, e.id) + ": " + coalesce(e.description, "")) AS all_descriptions
        ORDER BY final_score DESC
        LIMIT 3
        """
        
        context_parts = []
        raw_chunks = []
        try:
            async with self.driver.session(database=self.neo4j_db) as session:
                print(f"🔍 GraphRAG: Running True Hybrid Search (Context + Entity + Graph)...")
                result = await session.run(cypher, query_vector=query_vector)
                records = await result.data()
                print(f"📦 GraphRAG: True hybrid search returned {len(records)} record(s)")

                if records:
                    unique_facts = set()
                    unique_descriptions = set()
                    unique_chunks = []
                    
                    for r in records:
                        # Extract relationships found in the graph
                        if r.get('all_facts'):
                            for fact in r['all_facts']:
                                if isinstance(fact, list):
                                    for f in fact:
                                        if f:
                                            unique_facts.add(f)
                                elif isinstance(fact, str) and fact:
                                    unique_facts.add(fact)
                        
                        # Extract entity descriptions
                        if r.get('all_descriptions'):
                            for desc in r['all_descriptions']:
                                if isinstance(desc, list):
                                    for d in desc:
                                        if d and not d.endswith("No description available") and not d.endswith(": "):
                                            unique_descriptions.add(d)
                                elif isinstance(desc, str) and desc:
                                    if not desc.endswith("No description available") and not desc.endswith(": "):
                                        unique_descriptions.add(desc)
                        
                        # Extract the grounding context text
                        if r.get('chunk_text') and r.get('chunk_text') not in unique_chunks:
                            unique_chunks.append(r['chunk_text'])
                    
                    if unique_chunks:
                        context_parts.append("[GROUNDING CONTEXT]")
                        context_parts.extend(unique_chunks)
                        context_parts.append("") # Spacer
                        raw_chunks.extend(unique_chunks)

                    if unique_facts:
                        context_parts.append("[GRAPH RELATIONSHIPS]")
                        context_parts.extend(list(unique_facts))
                        context_parts.append("") # Spacer
                        raw_chunks.extend(list(unique_facts))
                        
                    if unique_descriptions:
                        context_parts.append("[ENTITY DESCRIPTIONS]")
                        context_parts.extend(list(unique_descriptions))
                        context_parts.append("") # Spacer
                        raw_chunks.extend(list(unique_descriptions))
                    
                else:
                    print(f"❌ GraphRAG: No matches found in vector indexes.")

        except Exception as e:
            print(f"⚠️ GraphRAG Retrieval error: {e}")
            
        if not context_parts:
            print(f"❌ GraphRAG: No context found for query='{query}'")
            return [] if as_list else ""
            
        if as_list:
            return raw_chunks
            
        final_context = "\n".join(context_parts)
        print(f"✅ GraphRAG: Retrieved Context Content:\n{'-'*40}\n{final_context}\n{'-'*40}")
        # Return full context, let LLM handle it, or cap at a much higher limit to avoid truncating grounding text
        return final_context[:12000]

    async def _flush_stop_frame(self, direction):
        if self.saved_stop_frame:
            print("🚦 GraphRAG: Flushing buffered Stop Frame")
            await self.push_frame(self.saved_stop_frame, direction)
            self.saved_stop_frame = None

    async def _flush_stop_frame_delayed(self, delay, direction):
        await asyncio.sleep(delay)
        if self.saved_stop_frame:
             print("⏰ GraphRAG: Timeout! Flushing buffered Stop Frame")
             await self._flush_stop_frame(direction)

    async def process_frame(self, frame: Frame, direction):
        frame_type = type(frame).__name__
        
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
                GraphRAGProcessor.LAST_QUERY = user_text
                GraphRAGProcessor.LAST_CONTEXT = context
                
                enriched_prompt = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{user_text}\n\nInstructions:\n1. Read the [CONTEXT] carefully.\n2. Find the exact answer to the [QUESTION] in the [CONTEXT].\n3. Provide a natural, conversational sentence as your answer.\n4. If the answer is not in the [CONTEXT], say: 'I'm sorry, I couldn't find that in the database.'\n"
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
                    GraphRAGProcessor.LAST_QUERY = latest_query
                    GraphRAGProcessor.LAST_CONTEXT = context
                    
                    enriched_prompt = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{latest_query}\n\nInstructions:\n1. Read the [CONTEXT] carefully.\n2. Find the exact answer to the [QUESTION] in the [CONTEXT].\n3. Provide a natural, conversational sentence as your answer.\n4. If the answer is not in the [CONTEXT], say: 'I'm sorry, I couldn't find that in the database.'\n"
                    user_messages[-1]["content"] = enriched_prompt
            
            await self.push_frame(frame, direction)
            await self._flush_stop_frame(direction)
            return

        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

class AnswerLoggerProcessor(FrameProcessor):
    """Intercepts LLM streams to log the full Answer to file."""
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
                        f"--- GraphRAG Session ---\n"
                        f"Retrieved Context:\n{GraphRAGProcessor.LAST_CONTEXT}\n\n"
                        f"User Question:\n{GraphRAGProcessor.LAST_QUERY}\n\n"
                        f"Final Answer:\n{self.current_answer}\n"
                        f"{'-'*30}\n\n"
                    )
                    f.write(log_entry)
                self.current_answer = ""
                
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
