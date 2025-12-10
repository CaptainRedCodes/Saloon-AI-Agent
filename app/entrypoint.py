from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentServer,AgentSession, Agent, room_io
from livekit.plugins import silero
from livekit.plugins.turn_detector.english import EnglishModel

from app.config.settings import settings
from app.agent import Assistant
from app.knowledge_base import KnowledgeManager

load_dotenv()

knowledge_manager = KnowledgeManager()
try:
    knowledge_manager.initialize()
except Exception as e:
    print("Error")
    raise

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt=settings.stt,
        llm=settings.llm,
        tts=settings.tts,
        vad=silero.VAD.load(),
        turn_detection=EnglishModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(ctx),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)