from __future__ import annotations

"""Convert transcript JSON to WebVTT with optional translation.

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


def wrap_text(text: str, width: int = 42) -> str:
    lines = textwrap.wrap(text, width=width) or [""]
    return "<br>".join(lines)


def get_async_client(api_key: str, base_url: str) -> Any:
    try:
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    except AttributeError:
        return openai.AsyncClient(api_key=api_key, base_url=base_url)


def load_segments(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = None
    # Common structure: {"output": {"segments": [...]}}
    if isinstance(data, dict):
        output = data.get("output")
        if isinstance(output, dict):
            segments = output.get("segments") or output.get("chunks")
        # Some APIs may return the list at the top level
        if segments is None:
            segments = data.get("segments") or data.get("chunks")
    if not isinstance(segments, list):
        raise ValueError("Invalid transcript JSON structure")
    return segments


def group_segments_by_paragraph(
    segments: Iterable[dict[str, Any]], max_chars: int = 500
) -> list[list[dict[str, Any]]]:
    """Group consecutive segments into paragraphs.

    A new group starts when a segment ends with a sentence terminator
    ('.', '!', '?') or when the accumulated character count exceeds
    ``max_chars``.
    """

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    length = 0
    for seg in segments:
        current.append(seg)
        text = seg.get("text", "")
        length += len(text)
        if text.strip().endswith((".", "!", "?")) or length >= max_chars:
            groups.append(current)
            current = []
            length = 0
    if current:
        groups.append(current)
    return groups


async def translate_segment(
    client: Any,
    text: str,
    model: str,
    language: str,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
) -> str:
    prompt = (
        "Translate the following English text into natural, fluent "
        f"{language}. Keep all line breaks and return exactly the same "
        "number of lines as the input without any extra text:\n" + text
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
    groups: Iterable[list[dict[str, Any]]],
    model: str,
    language: str,
    concurrency: int,
    logger: logging.Logger,
) -> list[str]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for grp in groups:
        text = "\n".join(seg.get("text", "") for seg in grp)
        tasks.append(
            asyncio.create_task(
                translate_segment(
                    client,
                    text,
                    model,
                    language,
                    semaphore,
                    logger,
                )
            )
        )
    translations: list[str] = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        translations.append(await coro)
    return translations


def write_vtt(
    segments: list[dict[str, Any]],
    path: str,
    font_color: str = "white",
    background_color: str = "transparent",
    font_size: float = 1.0,
) -> None:
    """Write segments to a WebVTT file with custom styling."""
    shadow = "0 0 3px #000, 0 0 6px #000"
    if font_color.lower() in {"black", "#000", "#000000"}:
        shadow = "0 0 3px #FFF, 0 0 6px #FFF"

    lines = ["::cue {", f"  color: {font_color};"]
    if background_color.lower() != "transparent" and background_color != "":
        lines.append(f"  background-color: {background_color};")
    lines.extend(
        [
            f"  font-size: {font_size}em;",
            "  font-weight: 600;",
            f"  text-shadow: {shadow};",
            "  line-height: 1.35;",
            '  font-family: "Helvetica Neue", Arial, sans-serif;',
            "}",
            "",
        ]
    )
    style = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        f.write("STYLE\n")
        f.write(style)
        f.write("\n")
        for idx, seg in enumerate(segments, 1):
            start, end = seg.get("timestamp", [0, 0])
            en = seg.get("text", "").strip()
            trans = seg.get("trans", "").strip()
            if trans:
                en_line, trans_line = wrap_and_align(en, trans)
            else:
                en_line = wrap_text(en)
                trans_line = None
            f.write(f"{idx}\n")
            f.write(
                f"{seconds_to_timestamp(float(start))} --> {seconds_to_timestamp(float(end))}\n"
            )
            if trans_line is not None:
                f.write(f"{en_line}\n{trans_line}\n\n")
            else:
                f.write(f"{en_line}\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert transcript JSON to WebVTT with optional translation"
    )
    parser.add_argument("input", help="Path to transcript JSON file")
    parser.add_argument("--model", help="Override OPENAI_MODEL")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent API calls")
    parser.add_argument(
        "--lang",
        help="Translate into this language (omit for no translation)",
    )
    parser.add_argument(
        "--font-color",
        default="white",
        help="Subtitle text color (default: white)",
    )
    parser.add_argument(
        "--background-color",
        default="transparent",
        help="Subtitle background color (default: transparent)",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=1.0,
        help="Font size in em units for subtitles",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output VTT file (default: <input>.vtt in current directory)",
    )
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
    groups = group_segments_by_paragraph(segments)
    client = get_async_client(api_key, base_url)
    if args.lang:
        logger.info(
            "Translating %d segments in %d groups into %s using model %s...",
            len(segments),
            len(groups),
            args.lang,
            model,
        )
        translations = await translate_all(
            client, groups, model, args.lang, args.concurrency, logger
        )
        for grp, trans in zip(groups, translations):
            lines = trans.splitlines()

            if len(lines) != len(grp):
                logger.warning(
                    "Translation line count mismatch: expected %d, got %d", 
                    len(grp), len(lines)
                )

            for seg, line in zip(grp, lines):
                seg["trans"] = line.strip()
    out_name = os.path.splitext(os.path.basename(args.input))[0] + ".vtt"
    out_path = args.output or os.path.join(os.getcwd(), out_name)
    write_vtt(
        segments,
        out_path,
        args.font_color,
        args.background_color,
        args.font_size,
    )
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
