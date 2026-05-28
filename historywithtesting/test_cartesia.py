import os
import asyncio
from dotenv import load_dotenv
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.frames.frames import TextFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameProcessor

load_dotenv(override=True)

class ResultLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        # We don't need to do anything, just log that we got something
        await super().process_frame(frame, direction)
        if not isinstance(frame, (TextFrame, EndFrame)):
            print(f"✅ Received frame: {type(frame)}")
        await self.push_frame(frame, direction)

async def test_cartesia():
    api_key = os.getenv("CARTESIA_API_KEY")
    voice_id = "71a7ad14-091c-4e8e-a314-022ece01c121"
    
    print(f"📡 Testing Cartesia with API Key: {api_key[:5]}...{api_key[-5:] if api_key else 'NONE'}")
    
    if not api_key:
        print("❌ ERROR: CARTESIA_API_KEY not found in .env!")
        return

    try:
        tts = CartesiaTTSService(api_key=api_key, voice_id=voice_id)
        res = ResultLogger()
        
        # Create a real (but minimal) pipeline
        pipeline = Pipeline([tts, res])
        task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
        
        print("🔊 Sending real phrase to Cartesia...")
        await task.queue_frames([TextFrame("Testing one two three."), EndFrame()])
        
        runner = PipelineRunner()
        await runner.run(task)
        
        print("\n✨ Test Successful! Cartesia is producing audio frames.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_cartesia())
