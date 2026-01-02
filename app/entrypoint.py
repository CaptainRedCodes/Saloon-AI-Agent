from livekit.agents import JobContext, WorkerOptions, cli, Agent, AgentSession
from livekit.plugins import silero
from livekit.agents.llm import FunctionTool, RawFunctionTool, ProviderTool
from app.models.salon_model import SalonUserData
from app.agent import Assistant
from app.config.settings import settings
from app.information import INSTRUCTIONS
from livekit.agents import AgentServer
from livekit.agents.job import JobProcess

server = AgentServer()



@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """
    SINGLE USER LiveKit Agent entrypoint
    """
    # Connect to the room
    await ctx.connect()
    

    userdata = SalonUserData()
    assistant_instance = Assistant(session=userdata, ctx=ctx)
    

    
    tools: list[FunctionTool | RawFunctionTool | ProviderTool] = [
        assistant_instance.get_current_date_and_time,
        assistant_instance.modify_booking_detail,
        assistant_instance.get_booking_summary,
        assistant_instance.get_salon_information,
        assistant_instance.check_availability,
        assistant_instance.request_help, #Making this MultiAgents in next update
        assistant_instance.collect_customer_information,
        assistant_instance.select_service,
        assistant_instance.schedule_appointment
    ]

    agent = Agent(
        instructions=INSTRUCTIONS ,
        tools=tools,
    )

    vad = ctx.proc.userdata.get("vad")
    if vad is None:
        vad = silero.VAD.load()
        ctx.proc.userdata["vad"] = vad


    session = AgentSession(
        vad=vad,
        stt=settings.stt,
        llm=settings.llm,
        tts=settings.tts,
    )
    
    # Start the session
    await session.start(agent=agent, room=ctx.room)
    
    # Optional: Generate initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly and ask how you can help them today."
    )



if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )