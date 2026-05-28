
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TextFrame, LLMMessagesFrame, LLMRunFrame
import google.generativeai as genai
from loguru import logger

class GeminiRAGProcessor(FrameProcessor):
    """Uses Gemini to retrieve context and generate response using RAG.
    
    This processor:
    1. Retrieves relevant context from knowledge base using Gemini
    2. Generates the final answer using that context
    3. Outputs TextFrame with the response
    """
    def __init__(self, model, knowledge_base=None):
        super().__init__()
        self.model = model
        self.knowledge_base = knowledge_base

    async def process_frame(self, frame, direction):
        try:
            await super().process_frame(frame, direction)

            # Handle TextFrame (transcribed text from STT)
            if isinstance(frame, TextFrame):
                logger.info(f"GeminiRAG: Processing TextFrame: {frame.text}")
                await self._handle_text_query(frame.text, direction)
            
            # Handle LLMMessagesFrame (from context aggregator)
            elif isinstance(frame, LLMMessagesFrame):
                logger.info("GeminiRAG: Processing LLMMessagesFrame")
                try:
                    logger.debug(f"GeminiRAG: Frame messages: {frame.messages}")
                    
                    # Get the latest user message
                    user_messages = [msg for msg in frame.messages if msg.get("role") == "user"]
                    if not user_messages:
                        logger.warning("GeminiRAG: No user messages found in frame")
                        await self.push_frame(frame, direction)
                        return
                    
                    latest_query = user_messages[-1].get("content", "")
                    logger.info(f"GeminiRAG: User query: '{latest_query}'")
                    
                    if latest_query:
                        await self._handle_text_query(latest_query, direction)
                    else:
                        logger.warning("GeminiRAG: Empty query in message")
                        await self.push_frame(TextFrame("I didn't understand that."), direction)
                        
                except Exception as e:
                    logger.error(f"GeminiRAG: Error processing LLMMessagesFrame: {e}", exc_info=True)
                    await self.push_frame(TextFrame("I encountered an error."), direction)
                    
            elif isinstance(frame, LLMRunFrame):
                # Pass through LLMRunFrame
                await self.push_frame(frame, direction)
            else:
                # Pass through other frame types silently
                await self.push_frame(frame, direction)
                
        except Exception as e:
            logger.error(f"GeminiRAG: Unexpected error in process_frame: {e}", exc_info=True)
            # Try to push an error response
            try:
                await self.push_frame(TextFrame("An error occurred processing your request."), direction)
            except Exception as push_error:
                logger.error(f"GeminiRAG: Failed to push error frame: {push_error}")

    async def _handle_text_query(self, query: str, direction):
        """Handle a text query and push response."""
        try:
            logger.info(f"GeminiRAG: Processing query: '{query}'")
            
            # Build prompt with knowledge base context
            if self.knowledge_base:
                prompt = f"""You are a helpful voice assistant. Answer questions concisely and conversationally.

Knowledge Base:
{self.knowledge_base}

User Question: {query}

Answer based on the knowledge base. If the answer is not in the knowledge base, answer from general knowledge."""
            else:
                prompt = f"""You are a helpful voice assistant. Answer questions concisely and conversationally.

User Question: {query}"""
            
            # Use Gemini to generate response with context
            logger.info("GeminiRAG: Calling Gemini for response generation...")
            response = self.model.generate_content(prompt)
            
            # Check if response is valid and has text
            if response and hasattr(response, 'text') and response.text:
                response_text = str(response.text).strip()
                logger.info(f"GeminiRAG: Generated response: {response_text[:100]}...")
                await self.push_frame(TextFrame(response_text), direction)
            elif response and hasattr(response, 'parts') and response.parts:
                # Extract text from response parts
                response_text = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
                
                if response_text:
                    logger.info(f"GeminiRAG: Generated response: {response_text[:100]}...")
                    await self.push_frame(TextFrame(response_text), direction)
                else:
                    logger.warning("GeminiRAG: Empty response text from Gemini")
                    await self.push_frame(TextFrame("I couldn't generate a response."), direction)
            else:
                logger.warning(f"GeminiRAG: Empty or invalid response from Gemini")
                await self.push_frame(TextFrame("I couldn't generate a response."), direction)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"GeminiRAG: Error in _handle_text_query: {error_msg}", exc_info=True)
            
            # Check for quota exceeded error
            if "429" in error_msg or "quota" in error_msg.lower() or "ResourceExhausted" in error_msg:
                await self.push_frame(TextFrame("I've exceeded my API quota. Please try again later."), direction)
            else:
                await self.push_frame(TextFrame("I encountered an error processing your request."), direction)
