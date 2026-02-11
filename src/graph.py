import asyncio
import base64
import json
import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

import aiofiles
from google import genai
from jinja2 import Environment, FileSystemLoader
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from openai import AsyncClient
from PIL import Image

from src.models import ImageEvalResponse, PlanResponse, UsageMetadata
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

    result_path: Path
    llm_model: str
    user_prompt: str
    gen_mode: Literal["init", "later"] = "init"
    enhanced_prompt: str
    user_image_path: Path
    init_image_path: Path
    images: list[str]
    job_path: Path
    plan_gen_duration_sec: float
    img_gen_duration_sec: asyncio.Queue[float]
    img_eval_duration_sec: asyncio.Queue[float]
    token_usage_queue: asyncio.Queue[UsageMetadata]
    specific_img_details: str


@dataclass
class RuntimeContext:
    """"""


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
        eval_schema = ImageEvalResponse.model_json_schema()
        prompt = await render_template_async(
            "plan.jinja",
            user_prompt=state["user_prompt"],
            eval_schema=json.dumps(eval_schema, indent=4),
        )
        img_data_url = await img_path_to_data_url(state["user_image_path"])
        async with openai_semaphore:
            t0 = time.perf_counter()
            response = await openai_client.responses.parse(
                model=PLANNER_MODEL,
                text_format=PlanResponse,
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
        await state["token_usage_queue"].put(usage)

        async with aiofiles.open(state["job_path"] / "plan.json", "w") as f:
            await f.write(response.output_parsed.model_dump_json(indent=4))

        return Send(
            "init_image_gen",
            {
                "enhanced_prompt": response.output_parsed.improved_prompt,
                "images": response.output_parsed.images,
                "plan_gen_duration_sec": plan_gen_seconds,
                "specific_img_details": (
                    response.output_parsed.images[0]
                    if response.output_parsed.images
                    else ""
                ),
            },
        )
    except Exception:
        logger.exception("Error occured during process planning.")


@dataclass
class ImgGenResponse:
    img_base64: str
    usage: UsageMetadata
    img_gen_seconds: float


async def google_img_gen(
    model: str, prompt: str, result_path: Path, image_path: Path
) -> ImgGenResponse:
    img = Image.open(image_path)
    async with google_semaphore:
        t0 = time.perf_counter()
        response = await gemini_client.aio.models.generate_content(
            model=model,
            contents=[prompt, img],
        )
        img_gen_seconds = time.perf_counter() - t0
        logger.info(f"Image was generated with '{model}'. Duration: {img_gen_seconds}s")
    usage = UsageMetadata.from_google_usage(model, response.usage_metadata)
    for part in response.parts:
        if part.text:
            text_path = result_path / f"{model}.txt"
            async with aiofiles.open(text_path, "w") as f:
                await f.write(part.text)
        elif part.inline_data:
            image_bytes = part.as_image().image_bytes
            img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return ImgGenResponse(
        img_base64=img_base64, usage=usage, img_gen_seconds=img_gen_seconds
    )


async def openai_img_gen(
    model: str, prompt: str, result_path: Path, image_path: Path
) -> ImgGenResponse:
    async with openai_semaphore:
        t0 = time.perf_counter()
        response = await openai_client.images.edit(
            model=model,
            prompt=prompt,
            image=open(image_path, "rb"),
        )
        img_gen_seconds = time.perf_counter() - t0
        logger.info(f"Image was generated with '{model}'. Duration: {img_gen_seconds}s")
    usage = UsageMetadata.from_openai_usage(model, response.usage)
    img_base64 = response.data[0].b64_json
    return ImgGenResponse(
        img_base64=img_base64, usage=usage, img_gen_seconds=img_gen_seconds
    )


async def image_gen(
    state: GraphState, context: RuntimeContext, gen_mode: Literal["init", "later"]
):
    """Geneates an image based on user's prompt and prefernce image."""
    try:
        if gen_mode == "init":
            image_path = state["user_image_path"]
            user_prompt = state["user_prompt"]
        else:
            image_path = state["init_image_path"]
            user_prompt = state["enhanced_prompt"]

        prompt = await render_template_async(
            "img_gen.jinja",
            user_prompt=user_prompt,
            specific_details=state["specific_img_details"],
        )
        model = state["llm_model"]
        if "gemini" in model:
            response = await google_img_gen(
                model=model,
                prompt=prompt,
                result_path=state["result_path"],
                image_path=image_path,
            )
        elif "gpt" in model:
            response = await openai_img_gen(
                model=model,
                prompt=prompt,
                result_path=state["result_path"],
                image_path=image_path,
            )
        else:
            raise NotImplementedError(f"Model '{model}' is not supported.")

        result_img_path = (
            state["result_path"] / f"{standardize_name(model)}_{uuid4()}.png"
        )
        async with aiofiles.open(result_img_path, "wb") as f:
            img_bytes = base64.b64decode(response.img_base64)
            await f.write(img_bytes)

        await state["token_usage_queue"].put(response.usage)
        await state["img_gen_duration_sec"].put(response.img_gen_seconds)

        return Command(goto="eval_image", update={})
    except Exception as e:
        logger.exception(f"Error occured during image generation")
        raise


async def eval_image(state: GraphState, context: RuntimeContext):
    """"""


async def final_per_image(state: GraphState, context: RuntimeContext):
    """"""


async def init_image_gen(state: GraphState, context: RuntimeContext):
    return await image_gen(state, context, "init")


async def later_image_gen(state: GraphState, context: RuntimeContext):
    return await image_gen(state, context, "later")


async def build_graph():
    graph_builder = StateGraph(GraphState, context_schema=RuntimeContext)
    graph_builder.add_node("plan", plan)
    graph_builder.add_node("init_image_gen", init_image_gen)
    graph_builder.add_node("later_image_gen", later_image_gen)
    graph_builder.add_node("eval_image", eval_image)
    graph_builder.add_node("final_per_image", final_per_image)

    graph_builder.add_edge(START, "plan")
    graph_builder.add_edge("plan", "init_image_gen")

    def fanout_after_init(state: GraphState):
        if len(state["images"]) == 1:
            return "final_per_image"

        return [
            Send(
                "later_image_gen",
                {"specific_img_details": details},
            )
            for details in state["images"]
        ]

    graph_builder.add_conditional_edges(
        "init_image_gen", fanout_after_init, ["later_image_gen", "final_per_image"]
    )
    graph_builder.add_edge("later_image_gen", "eval_image")

    # TODO should "final" wait for all "eval"s?
    graph_builder.add_edge("eval_image", "final_per_image")
    graph_builder.add_edge("final_per_image", END)

    return graph_builder.compile()
