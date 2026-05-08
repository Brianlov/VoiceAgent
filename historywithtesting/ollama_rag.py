
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TextFrame, LLMMessagesFrame, LLMRunFrame
import requests
from loguru import logger

class OllamaRAGProcessor(FrameProcessor):
    """Uses Ollama to retrieve context and generate response using RAG.
    
    This processor:
    1. Retrieves relevant context from knowledge base
    2. Generates the final answer using Ollama
    3. Outputs TextFrame with the response
    """
    def __init__(self, base_url="http://localhost:11434/v1", model="qwen2:1.5b", knowledge_base=None):
        super().__init__()
        self.base_url = base_url
        self.model = model
        self.knowledge_base = knowledge_base
        self.timeout = 30  # 30 seconds timeout

    async def process_frame(self, frame, direction):
        try:
            await super().process_frame(frame, direction)

            # Handle TextFrame (transcribed text from STT)
            if isinstance(frame, TextFrame):
                logger.info(f"OllamaRAG: Processing TextFrame: {frame.text}")
                await self._handle_text_query(frame.text, direction)
            
            # Handle LLMMessagesFrame (from context aggregator)
            elif isinstance(frame, LLMMessagesFrame):
                logger.info("OllamaRAG: Processing LLMMessagesFrame")
                try:
                    logger.debug(f"OllamaRAG: Frame messages: {frame.messages}")
                    
                    # Get the latest user message
                    user_messages = [msg for msg in frame.messages if msg.get("role") == "user"]
                    if not user_messages:
                        logger.warning("OllamaRAG: No user messages found in frame")
                        await self.push_frame(frame, direction)
                        return
                    
                    latest_query = user_messages[-1].get("content", "")
                    logger.info(f"OllamaRAG: User query: '{latest_query}'")
                    
                    if latest_query:
                        await self._handle_text_query(latest_query, direction)
                    else:
                        logger.warning("OllamaRAG: Empty query in message")
                        await self.push_frame(TextFrame("I didn't understand that."), direction)
                        
                except Exception as e:
                    logger.error(f"OllamaRAG: Error processing LLMMessagesFrame: {e}", exc_info=True)
                    await self.push_frame(TextFrame("I encountered an error."), direction)
                    
            elif isinstance(frame, LLMRunFrame):
                # Pass through LLMRunFrame
                await self.push_frame(frame, direction)
            else:
                # Pass through other frame types silently
                await self.push_frame(frame, direction)
                
        except Exception as e:
            logger.error(f"OllamaRAG: Unexpected error in process_frame: {e}", exc_info=True)
            # Try to push an error response
            try:
                await self.push_frame(TextFrame("An error occurred processing your request."), direction)
            except Exception as push_error:
                logger.error(f"OllamaRAG: Failed to push error frame: {push_error}")

    async def _handle_text_query(self, query: str, direction):
        """Handle a text query and push response."""
        try:
            logger.info(f"OllamaRAG: Processing query: '{query}'")
            
            # Build prompt with knowledge base context
            if self.knowledge_base:
                prompt = f"""You are a helpful voice assistant. Answer questions concisely and conversationally.

Knowledge Base:
{self.knowledge_base}

User Question: {query}

Answer based on the knowledge base. If the answer is not in the knowledge base, answer from general knowledge. Keep your response brief and conversational."""
            else:
                prompt = f"""You are a helpful voice assistant. Answer questions concisely and conversationally.

User Question: {query}

Keep your response brief and conversational."""
            
            # Use Ollama to generate response with context
            logger.info(f"OllamaRAG: Calling Ollama ({self.model}) for response generation...")
            
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "stream": False,
                "temperature": 0.7,
            }
            
            response = requests.post(url, json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                if response_text:
                    logger.info(f"OllamaRAG: Generated response: {response_text[:100]}...")
                    await self.push_frame(TextFrame(response_text), direction)
                else:
                    logger.warning("OllamaRAG: Empty response from Ollama")
                    await self.push_frame(TextFrame("I couldn't generate a response."), direction)
            else:
                logger.error(f"OllamaRAG: Ollama returned status {response.status_code}: {response.text}")
                await self.push_frame(TextFrame("I encountered an error with the language model."), direction)
                
        except requests.Timeout:
            logger.error(f"OllamaRAG: Request timed out after {self.timeout} seconds")
            await self.push_frame(TextFrame("The response took too long. Please try again."), direction)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"OllamaRAG: Error in _handle_text_query: {error_msg}", exc_info=True)
            await self.push_frame(TextFrame("I encountered an error processing your request."), direction)
