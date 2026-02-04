import asyncio
import base64
import json
import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import aiofiles
from google import genai
from jinja2 import Environment, FileSystemLoader
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from openai import AsyncClient

from src.models import ImageJudgeResponse, PromptEnhancementResponse, UsageMetadata
from src.settings import settings

logger = logging.getLogger(__name__)

SEMAPHORE_VAL = 5
PROMPTS_PATH = Path("prompts")
EVAL_MODEL = "gpt-5-mini"
PLANNER_MODEL = "gpt-5-mini"

env = Environment(loader=FileSystemLoader("prompts"))
openai_semaphore = asyncio.Semaphore(SEMAPHORE_VAL)
google_semaphore = asyncio.Semaphore(SEMAPHORE_VAL)
openai_client = AsyncClient(api_key=settings.OPENAI_API_KEY)
gemini_client = genai.Client()


def standardize_name(name: str) -> str:
    if "/" in name:
        return name.replace("/", "__")
    return name


class GraphState(TypedDict):
    """Graph state of the image generation graph."""

    user_prompt: str
    enhanced_prompt: str
    user_image_path: Path
    init_image_path: Path
    images: list
    job_path: Path
    plan_gen_duration_sec: float
    img_gen_duration_sec: asyncio.Queue[float]
    img_eval_duration_sec: asyncio.Queue[float]


@dataclass
class RuntimeContext:
    """"""

    token_usage_queue: asyncio.Queue[UsageMetadata]


async def img_path_to_data_url(img_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(img_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type")

    async with aiofiles.open(img_path, "rb") as f:
        content = await f.read()
        encoded = base64.b64encode(content).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"


async def render_template_async(template_name: str, **context) -> str:
    template_path = PROMPTS_PATH / template_name
    async with aiofiles.open(template_path, "r", encoding="utf-8") as f:
        template_str = await f.read()

    template = env.from_string(template_str)
    return template.render(**context)


async def plan(state: GraphState, runtime: RuntimeContext):
    """Plan the further actions."""
    try:
        eval_schema = ImageJudgeResponse.model_json_schema()
        async with openai_semaphore:
            t0 = time.perf_counter()
            prompt = await render_template_async(
                "prompt_enhancer.jinja",
                user_prompt=state["user_prompt"],
                eval_schema=json.dumps(eval_schema, indent=4),
            )
            img_data_url = await img_path_to_data_url(state["user_image_path"])
            response = await openai_client.responses.parse(
                model=PLANNER_MODEL,
                text_format=PromptEnhancementResponse,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": img_data_url},
                        ],
                    }
                ],
            )
            plan_gen_seconds = time.perf_counter() - t0
            logger.info(f"Interrier plan was generated. Duration: {plan_gen_seconds}s")

            usage = UsageMetadata.from_openai_usage(response.usage)
            await runtime.token_usage_queue.put(usage)

            async with aiofiles.open(state["job_path"] / "plan.json", "w") as f:
                await f.write(response.output_parsed.model_dump_json(indent=4))

            return Send(
                "image_gen",
                {
                    "enhanced_prompt": response.output_parsed.improved_prompt,
                    "images": response.output_parsed.images,
                    "plan_gen_duration_sec": plan_gen_seconds,
                },
            )
    except Exception:
        logger.exception("Error occured during process planning.")


async def image_gen(state: GraphState, context: RuntimeContext):
    """"""


async def eval(state: GraphState, context: RuntimeContext):
    """"""


async def final(state: GraphState, context: RuntimeContext):
    """"""


async def build_graph():
    graph_builder = StateGraph(GraphState, context_schema=RuntimeContext)
    graph_builder.add_node("plan", plan)
    graph_builder.add_node("image_gen", image_gen)
    graph_builder.add_node("eval", eval)
    graph_builder.add_node("final", final)

    graph_builder.add_edge(START, "plan")
    graph_builder.add_edge("plan", "image_gen")

    def fanout_after_first(state: GraphState):
        return [
            Send("image_gen", {"specific_desciption": desciption})
            for desciption in state["images"]
        ]

    graph_builder.add_conditional_edges("image_gen", fanout_after_first, ["image_gen"])
    graph_builder.add_edge("image_gen", "eval")
    graph_builder.add_edge("eval", "final")
    graph_builder.add_edge("final", END)

    return graph_builder.compile()
