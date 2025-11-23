import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json
from livekit.agents import AgentTask, function_tool, RunContext
from livekit.agents import Agent






@dataclass
class CoffeeOrderResult:
    drinkType: str | None = None
    size: str | None = None
    milk: str | None = None
    extras: list[str] = field(default_factory=list)
    name: str | None = None


ORDERS_DIR = Path(__file__).parent.parent / "orders"
ORDERS_DIR.mkdir(exist_ok=True)

def save_order_to_json(order: CoffeeOrderResult) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ORDERS_DIR / f"order_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(order), f, indent=2, ensure_ascii=False)
    return str(path)


class CollectCoffeeOrderTask(AgentTask[CoffeeOrderResult]):
     def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            instructions="""
You are a friendly barista at Nothing Before Coffee Café.

Your job is to take a coffee order and fill this object:

{
  "drinkType": "string",
  "size": "string",
  "milk": "string",
  "extras": ["string"],
  "name": "string"
}

Guidelines:
- Ask questions conversationally until you know ALL 5 fields.
- You can collect them in any order.
- Clarify politely if anything is ambiguous.
- When the user changes their mind, update the relevant field.
- When you're sure everything is complete, call the appropriate tools to record the values.
- Do NOT invent details if the user doesn't say them; ask.

""",
            chat_ctx=chat_ctx,
        )
        self._order: dict[str, object] = {}

     def _check_completion(self):
        required_keys = {"drinkType", "size", "milk", "extras", "name"}
        if required_keys.issubset(self._order.keys()):
            # ensure extras is always a list
            extras = self._order.get("extras") or []
            if isinstance(extras, str):
                extras = [extras]

            result = CoffeeOrderResult(
                drinkType=str(self._order["drinkType"]),
                size=str(self._order["size"]),
                milk=str(self._order["milk"]),
                extras=list(extras),
                name=str(self._order["name"]),
            )
            self.complete(result)
        else:
            # keep going
            self.session.generate_reply(
                instructions="Continue asking brief follow-up questions to collect any missing fields."
            )

     @function_tool()
     async def set_drink_type(self, context: RunContext, drink_type: str):
        """Call this when you know the drink type (e.g. latte, cappuccino, cold brew, americano, mocha)."""
        self._order["drinkType"] = drink_type
        self._check_completion()

     @function_tool()
     async def set_size(self, context: RunContext, size: str):
        """Call this when you know the drink size. Use values like 'small', 'medium', or 'large'."""
        self._order["size"] = size
        self._check_completion()

     @function_tool()
     async def set_milk(self, context: RunContext, milk: str):
        """Call this when you know the milk type (e.g. regular, skim, oat, almond, soy)."""
        self._order["milk"] = milk
        self._order.setdefault("extras", [])
        self._check_completion()

     @function_tool()
     async def add_extra(self, context: RunContext, extra: str):
        """Call this to add an extra like 'extra shot', 'caramel syrup', 'whipped cream'. Can be called multiple times."""
        extras = self._order.setdefault("extras", [])
        if extra not in extras:
            extras.append(extra)
        self._check_completion()

     @function_tool()
     async def set_name(self, context: RunContext, name: str):
        """Call this when you know the customer's name for the cup."""
        self._order["name"] = name
        self._check_completion()



class CoffeeBaristaAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a warm, friendly barista at Nothing Before Coffee Café.
Greet the customer, chat briefly, and then guide them through placing a coffee order.
""",
        )

    async def on_enter(self) -> None:
        # Start the order task; this will control the conversation until done
        order_result: CoffeeOrderResult = await CollectCoffeeOrderTask(
            chat_ctx=self.chat_ctx
        )

        # Save to JSON file
        path = save_order_to_json(order_result)

        # Speak a neat summary back to the user
        extras_str = ", ".join(order_result.extras) if order_result.extras else "no extras"
        summary = (
            f"Thanks {order_result.name}! "
            f"So that's a {order_result.size} {order_result.drinkType} "
            f"with {order_result.milk} milk and {extras_str}. "
            f"Your order has been placed."
        )

        await self.session.generate_reply(
            instructions=f"Speak this summary to the user: {summary}"
        )

        # (optional) you can also mention that the order was saved on the system
        print(f"[BARISTA] Order saved to: {path}")


from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    # function_tool,
    # RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def barista_agent(ctx: JobContext) -> None:
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=CoffeeBaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

async def entrypoint(ctx: JobContext):
    await barista_agent(ctx)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
