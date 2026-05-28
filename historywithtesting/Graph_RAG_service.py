import os
import asyncio
import re
from neo4j import AsyncGraphDatabase
import logging
from typing import List

from pipecat.frames.frames import TextFrame, Frame, InterimTranscriptionFrame, UserStoppedSpeakingFrame, StartFrame, TranscriptionFrame, LLMMessagesAppendFrame, LLMMessagesFrame, LLMContextFrame
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

    async def retrieve(self, query: str) -> str:
        print(f"🚨 RETRIEVE CALLED with: '{query}'")
        if not self._initialized:
            await self._initialize_graph()
            
        if not self._initialized:
             return ""

        STOPWORDS = {
            'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their', 'them',
            'first', 'last', 'open', 'opened', 'begin', 'began', 'start', 'started',
            'state', 'year', 'time', 'many', 'much', 'more', 'most', 'some', 'any',
            'also', 'just', 'than', 'then', 'there', 'here', 'about', 'into', 'type',
            'changed', 'handling', 'result', 'stated', 'became', 'called', 'used',
            'original', 'originator', 'developed', 'basis', 'based', 'using'
        }

        query_lower = query.lower()
        words_list = re.findall(r'\b\w+\b', query_lower)
        words = set(words_list)
        
        # 1. Base keywords (>3 chars, no stopwords)
        keywords = [w for w in words if len(w) > 3 and w not in STOPWORDS]
        
        # 2. Phrase matching: Find 2-word and 3-word combinations that might be Entities
        # e.g. "Virgin Mary", "Main Building"
        phrases = []
        for i in range(len(words_list) - 1):
            bi_gram = f"{words_list[i]} {words_list[i+1]}"
            phrases.append(bi_gram)
            if i < len(words_list) - 2:
                tri_gram = f"{words_list[i]} {words_list[i+1]} {words_list[i+2]}"
                phrases.append(tri_gram)
        
        # We only care about phrases that aren't just stopwords
        for p in phrases:
            if any(len(w) > 3 for w in p.split()):
                keywords.append(p)

        if not keywords:
            keywords = [w for w in words if len(w) > 3]
        
        if not keywords:
            keywords = [query_lower]

        print(f"🔑 GraphRAG: Keywords (with Phrases) extracted: {keywords}")

        # Require multiple keywords to co-occur in the SAME context paragraph.
        # Prevents generic terms (e.g. 'france') from pulling unrelated contexts.
        # Dynamic Min Match: Require higher agreement for longer queries
        # For 5+ keywords, require at least 3 to match to prevent 'France' noise.
        num_kws = len(keywords)
        if num_kws >= 6:
            min_match = 3
        elif num_kws >= 3:
            min_match = 2
        else:
            min_match = 1
            
        weighted_kws = []
        anchors = []
        for kw in keywords:
            boost = 1.0
            is_anchor = False
            if " " in kw: boost = 10.0 # Phrase boost
            if re.match(r'^\d{4}$', kw): 
                boost = 25.0 # Massive boost for years
                is_anchor = True
            
            if kw.lower() in ["lourdes", "bernadette"]:
                boost = 20.0
                is_anchor = True
                
            weighted_kws.append({"kw": kw, "boost": boost, "is_anchor": is_anchor})
            if is_anchor: anchors.append(kw.lower())

        # Cypher: Anchor-Aware Ranking
        # Chunks missing the 'Anchor' words get penalized
        cypher = """
        UNWIND $weighted_kws AS item
        WITH item.kw AS kw, item.boost AS boost, item.is_anchor AS is_anchor
        MATCH (e:Entity)
        WHERE toLower(e.name) =~ ("(?i).*\\\\b" + kw + "\\\\b.*")
        
        // Calculate Specificity (Inverse Frequency)
        OPTIONAL MATCH (e)-[m:MENTIONED_IN]->()
        WITH e, kw, boost, is_anchor, count(m) AS global_mentions
        WITH e, kw, boost, is_anchor, (100.0 / (global_mentions + 5.0)) AS specificity_weight
        
        // Apply Phrase/Anchor Boost
        WITH collect(DISTINCT {node: e, weight: specificity_weight * boost, is_anchor: is_anchor}) AS matched_data
        WITH [item in matched_data | item.node] AS all_matched_entities, matched_data
        
        // 1. Find Direct Relationships & Discovery (Expand from matched entities)
        OPTIONAL MATCH (e1)-[r]->(e2)
        WHERE e1 IN all_matched_entities
          AND type(r) <> "MENTIONED_IN"
          
        WITH matched_data, all_matched_entities, e1, r, e2,
             (case when e2 IN all_matched_entities then 2.0 else 1.0 end) AS rel_quality
        
        ORDER BY rel_quality DESC
        LIMIT 15
        
        WITH matched_data, all_matched_entities,
             collect(DISTINCT coalesce(e1.name, e1.id) + "--" + type(r) + "--" + coalesce(e2.name, e2.id)) AS facts
        
        // 2. Find Context Chunks (Grounding)
        OPTIONAL MATCH (node:Entity)-[:MENTIONED_IN]->(c:Chunk)
        WHERE node IN all_matched_entities
        
        // Calculate Weighted Score for the Chunks
        WITH facts, c, node, matched_data
        UNWIND matched_data AS data
        WITH facts, c, node, data WHERE data.node = node
        
        WITH facts, c, sum(data.weight) AS base_weighted_score, 
             count(DISTINCT node) AS unique_matches,
             collect(DISTINCT data.is_anchor) AS found_anchors
             
        // Anchor Penalty: If query has anchors but chunk doesn't have them matched, slash score
        WITH facts, c, base_weighted_score, unique_matches,
             (case when true IN found_anchors then 1.0 else 0.1 end) AS anchor_modifier,
             (case when unique_matches >= 3 then 50.0 when unique_matches >= 2 then 10.0 else 1.0 end) AS intersection_multiplier
             
        WITH facts, c, (base_weighted_score * intersection_multiplier * anchor_modifier) AS final_weighted_score, unique_matches
        
        WHERE c is NOT NULL AND unique_matches >= $min_match
        RETURN facts, c.text AS chunk_text, final_weighted_score AS weighted_score, unique_matches
        ORDER BY final_weighted_score DESC
        LIMIT 3
        """
        
        context_parts = []
        try:
            async with self.driver.session(database=self.neo4j_db) as session:
                # Primary Graph-Facts Search
                print(f"🗄️ GraphRAG: Running phrase-aware Cypher with {len(weighted_kws)} weighted keywords...")
                result = await session.run(cypher, weighted_kws=weighted_kws, min_match=min_match)
                records = await result.data()
                print(f"📦 GraphRAG: Primary search returned {len(records)} record(s)")

                if records:
                    # RELATIONSHIPS FIRST (As requested: 'Just relationship')
                    unique_relationships = set()
                    for r in records:
                        if r.get('facts'):
                            for fact in r['facts']:
                                unique_relationships.add(fact)
                    
                    if unique_relationships:
                        context_parts.append("[DIRECT GRAPH RELATIONSHIPS]")
                        context_parts.extend(list(unique_relationships))
                        context_parts.append("") # Spacer
                    
                    # GROUNDING TEXT (Only if highly relevant, else drop entity facts)
                    relevant_chunks = []
                    for r in records:
                        # If a chunk matches multiple distinct keywords (intersection >= 2), 
                        # it's likely high-quality context, not just a generic entity fact.
                        if r.get('chunk_text') and r.get('unique_matches', 0) >= 2:
                             relevant_chunks.append(r['chunk_text'])
                    
                    if relevant_chunks:
                        context_parts.append("[GROUNDING CONTEXT]")
                        context_parts.extend(relevant_chunks)
                else:
                    # --- Diagnose: check if entities even exist ---
                    diag = await session.run(
                        "MATCH (e:Entity) RETURN e.id AS id LIMIT 10"
                    )
                    diag_records = await diag.data()
                    sample_ids = [r['id'] for r in diag_records]
                    print(f"🔎 GraphRAG: Sample entity IDs in DB: {sample_ids}")

                    # --- Diagnose: check relationship types ---
                    rel_diag = await session.run(
                        "MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel_type LIMIT 20"
                    )
                    rel_records = await rel_diag.data()
                    rel_types = [r['rel_type'] for r in rel_records]
                    print(f"🔗 GraphRAG: Relationship types in DB: {rel_types}")

                    # --- Diagnose: count nodes per label ---
                    label_diag = await session.run(
                        "CALL db.labels() YIELD label "
                        "CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) AS cnt', {}) "
                        "YIELD value RETURN label, value.cnt AS count"
                    )
                    try:
                        label_records = await label_diag.data()
                        print(f"🏷️ GraphRAG: Node counts by label: {label_records}")
                    except Exception:
                        # apoc may not be installed; fall back to simple counts
                        cnt_diag = await session.run(
                            "MATCH (n) RETURN labels(n) AS lbl, count(n) AS cnt"
                        )
                        cnt_records = await cnt_diag.data()
                        print(f"🏷️ GraphRAG: Node counts: {cnt_records}")

                    # --- Fallback: Broad search matching any entity name ---
                    # --- Fallback: Direct Full-Text Search on Chunk Content ---
                    # Use this when entity matching is too noisy or found nothing
                    # Also applying phrase boost to fallback
                    print(f"🔄 GraphRAG: Trying weighted Full-Text search on Chunk.text...")
                    fallback_cypher = """
                        UNWIND $weighted_kws AS item
                        WITH item.kw AS kw, item.boost AS boost
                        MATCH (c:Chunk)
                        WHERE toLower(c.text) CONTAINS kw
                        WITH c, count(DISTINCT kw) AS text_match_count, sum(boost) AS weighted_match_score
                        WHERE text_match_count >= $min_match
                        RETURN c.text AS text, weighted_match_score
                        ORDER BY weighted_match_score DESC
                        LIMIT 3
                    """
                    fb_result = await session.run(fallback_cypher, weighted_kws=weighted_kws, min_match=min_match)
                    fb_records = await fb_result.data()
                    print(f"🔄 GraphRAG: Full-Text Fallback returned {len(fb_records)} record(s)")
                    for r in fb_records:
                        if r.get('text'):
                            context_parts.append(r['text'])

                    if not fb_records:
                        print(f"❌ GraphRAG: No knowledge found in text or entities for {keywords}.")

        except Exception as e:
            print(f"⚠️ GraphRAG Retrieval error: {e}")
            
        if not context_parts:
            print(f"❌ GraphRAG: No context found for query='{query}'")
            return ""
            
        print(f"✅ GraphRAG: Returning {len(context_parts)} context chunk(s)")
        return " ".join(context_parts)[:3500]  # cap to limit LLM token load



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
        
        if "AudioRawFrame" not in frame_type:
            pass  # print(f"📨 GraphRAG RECV: {frame_type}")

        if isinstance(frame, StartFrame):
            print("🚀 GraphRAG: Initializing and forwarding StartFrame")
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterimTranscriptionFrame):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return
            
        if isinstance(frame, UserStoppedSpeakingFrame):
            print("🛑 GraphRAG: Buffering Stop Frame (waiting for text)")
            self.saved_stop_frame = frame
            asyncio.create_task(self._flush_stop_frame_delayed(0.5, direction))
            return

        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text
            if user_text.strip():
                print(f"🔍 GraphRAG: Searching graph for context related to: '{user_text}'")
                context = await self.retrieve(user_text)
                
                GraphRAGProcessor.LAST_QUERY = user_text
                GraphRAGProcessor.LAST_CONTEXT = context
                
                logger.info(f"[RAG OUTPUT]: {context}")
                
                print(f"💡 GraphRAG: Rewriting query...")
                prompt = f"""[CONTEXT]\n{context}\n\n[QUESTION]\n{user_text}"""
                frame.text = prompt
            print(f"🔄 GraphRAG: Pushing TranscriptionFrame with text: {frame.text[:50]}...")
            await self.push_frame(frame, direction)
            await self._flush_stop_frame(direction)
            from pipecat.frames.frames import LLMRunFrame
            await self.push_frame(LLMRunFrame(), direction)
            return

        if isinstance(frame, LLMMessagesAppendFrame):
            if frame.messages and frame.messages[-1].get("role") == "user":
                user_text = frame.messages[-1].get("content", "")
                if isinstance(user_text, str) and user_text.strip():
                    print(f"🔍 GraphRAG: Searching graph for context related to text input: '{user_text}'")
                    context = await self.retrieve(user_text)
                    
                    GraphRAGProcessor.LAST_QUERY = user_text
                    GraphRAGProcessor.LAST_CONTEXT = context
                    
                    logger.info(f"[RAG OUTPUT]: {context}")
                    
                    print(f"💡 GraphRAG: Rewriting query...")
                    prompt = f"""[CONTEXT]\n{context}\n\n[QUESTION]\n{user_text}"""
                    frame.messages[-1]["content"] = prompt
            print(f"🔄 GraphRAG: Pushing LLMMessagesAppendFrame...")
            await self.push_frame(frame, direction)
            await self._flush_stop_frame(direction)
            from pipecat.frames.frames import LLMRunFrame
            await self.push_frame(LLMRunFrame(), direction)
            return

        if isinstance(frame, TextFrame):
            await asyncio.sleep(0)
            
            user_text = frame.text
            if user_text:
                print(f"🔍 GraphRAG: Searching graph for context related to: '{user_text}'")
                context = await self.retrieve(user_text)
                
                GraphRAGProcessor.LAST_QUERY = user_text
                GraphRAGProcessor.LAST_CONTEXT = context
                
                logger.info(f"[RAG OUTPUT]: {context}")
                
                print(f"💡 GraphRAG: Rewriting query...")
                prompt = f"""[CONTEXT]\n{context}\n\n[QUESTION]\n{user_text}"""
                frame.text = prompt
            
            print(f"🔄 GraphRAG: Pushing frame with text: {frame.text[:50]}...")
            await self.push_frame(frame, direction)
            
            await self._flush_stop_frame(direction)
            return

        if isinstance(frame, LLMContextFrame):
            messages = frame.context.messages
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            
            if user_messages:
                latest_query = user_messages[-1].get("content", "")
                if isinstance(latest_query, str) and latest_query.strip():
                    print(f"🔍 GraphRAG [LLMContextFrame]: Searching for: '{latest_query}'")
                    context = await self.retrieve(latest_query)
                    
                    GraphRAGProcessor.LAST_QUERY = latest_query
                    GraphRAGProcessor.LAST_CONTEXT = context
                    
                    logger.info(f"[RAG OUTPUT]: {context}")
                    
                    print(f"📄 GraphRAG CONTEXT:\n{'='*60}\n{context}\n{'='*60}")
                    
                    enriched_prompt = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{latest_query}"
                    enriched_prompt = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{latest_query}\n\nAnswer ONLY using the provided [CONTEXT]. Extract the EXACT answer from the text. If the information is not present in the [CONTEXT] or the context is empty, say: 'I'm sorry, I couldn't find any specific records for that in the database.'"
                    user_messages[-1]["content"] = enriched_prompt
                    print(f"💡 GraphRAG: Injected context into LLMContextFrame")
            
            await self.push_frame(frame, direction)
            await self._flush_stop_frame(direction)
            return


            # 1. Extract the latest user query from the messages list
            messages = frame.messages.copy()
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            
            if not user_messages:
                await self.push_frame(frame, direction)
                return

            latest_query = user_messages[-1].get("content", "")
            if not isinstance(latest_query, str) or not latest_query.strip():
                await self.push_frame(frame, direction)
                return

            # 2. Search Neo4j Graph
            print(f"🔍 GraphRAG: Searching graph for context related to: '{latest_query}'")
            context = await self.retrieve(latest_query)
            
            GraphRAGProcessor.LAST_QUERY = latest_query
            GraphRAGProcessor.LAST_CONTEXT = context
            
            logger.info(f"[RAG OUTPUT]: {context}")
            
            print(f"💡 GraphRAG: Rewriting message with context...")
            enriched_prompt = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{latest_query}"
            
            # Replace the user content instead of overriding system prompt
            user_messages[-1]["content"] = enriched_prompt

            print(f"🔄 GraphRAG: Pushing enriched LLMMessagesFrame...")
            await self.push_frame(LLMMessagesFrame(messages), direction)
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
        elif frame_type == "LLMFullResponseEndFrame" or frame_type == "TTSStoppedFrame" or frame_type == "UserStartedSpeakingFrame" or frame_type == "UserStoppedSpeakingFrame":
            if self.current_answer.strip():
                with open("graphrag_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"GraphTranversal:\ndata retrrieval:\n\"{GraphRAGProcessor.LAST_CONTEXT}\"\nCombine with the questions\nQuestions:\n{GraphRAGProcessor.LAST_QUERY}\nAnswer:\n{self.current_answer}\n\n")
                self.current_answer = ""
                
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
