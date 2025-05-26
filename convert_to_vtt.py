from __future__ import annotations

"""Convert transcript JSON to bilingual WebVTT.

Dependencies:
    pip install openai>=1.0.0 tqdm
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import textwrap
from typing import Any, Iterable, Tuple

import openai
from tqdm import tqdm


def seconds_to_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def wrap_and_align(en: str, cn: str, width: int = 42) -> Tuple[str, str]:
    en_lines = textwrap.wrap(en, width=width) or [""]
    cn_lines = textwrap.wrap(cn, width=width) or [""]
    max_len = max(len(en_lines), len(cn_lines))
    en_lines += [""] * (max_len - len(en_lines))
    cn_lines += [""] * (max_len - len(cn_lines))
    return "<br>".join(en_lines), "<br>".join(cn_lines)


def get_async_client(api_key: str, base_url: str) -> Any:
    try:
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    except AttributeError:
        return openai.AsyncClient(api_key=api_key, base_url=base_url)


def load_segments(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        segments = data["output"]["segments"]
    except Exception as exc:
        raise ValueError("Invalid transcript JSON structure") from exc
    if not isinstance(segments, list):
        raise ValueError("segments should be a list")
    return segments


async def translate_segment(
    client: Any,
    text: str,
    model: str,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
) -> str:
    prompt = (
        "Translate the following English sentence into natural, fluent Simplified Chinese and return ONLY the translation with no additional text:\n"
        + text
    )
    for attempt in range(3):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
            content = resp.choices[0].message.content.strip()
            return content
        except Exception as exc:  # type: ignore
            wait = 2 ** attempt
            logger.error("Translation attempt %d failed: %s", attempt + 1, exc)
            if attempt == 2:
                raise
            await asyncio.sleep(wait)
    return ""


async def translate_all(
    client: Any,
    segments: Iterable[dict[str, Any]],
    model: str,
    concurrency: int,
    logger: logging.Logger,
) -> list[str]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            translate_segment(client, seg.get("text", ""), model, semaphore, logger)
        )
        for seg in segments
    ]
    translations: list[str] = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        translations.append(await coro)
    return translations


def write_vtt(segments: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for idx, seg in enumerate(segments, 1):
            start, end = seg.get("timestamp", [0, 0])
            en = seg.get("text", "").strip()
            cn = seg.get("cn", "").strip()
            en_line, cn_line = wrap_and_align(en, cn)
            f.write(f"{idx}\n")
            f.write(f"{seconds_to_timestamp(float(start))} --> {seconds_to_timestamp(float(end))}\n")
            f.write(f"{en_line}\n{cn_line}\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert transcript JSON to bilingual WebVTT")
    parser.add_argument("input", help="Path to transcript JSON file")
    parser.add_argument("--model", help="Override OPENAI_MODEL")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent API calls")
    return parser.parse_args()


async def async_main(args: argparse.Namespace, logger: logging.Logger) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    model = args.model or os.getenv("OPENAI_MODEL")
    if not model:
        logger.error("OpenAI model not specified via OPENAI_MODEL or --model")
        sys.exit(1)

    segments = load_segments(args.input)
    client = get_async_client(api_key, base_url)
    logger.info("Translating %d segments using model %s...", len(segments), model)
    translations = await translate_all(client, segments, model, args.concurrency, logger)
    for seg, trans in zip(segments, translations):
        seg["cn"] = trans
    out_name = os.path.splitext(os.path.basename(args.input))[0] + ".vtt"
    out_path = os.path.join(os.getcwd(), out_name)
    write_vtt(segments, out_path)
    logger.info("Wrote %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("convert_to_vtt")
    args = parse_args()
    try:
        asyncio.run(async_main(args, logger))
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
