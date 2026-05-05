#!/usr/bin/env python3
"""Common helpers for generating process-supervision traces over produced datasets."""

from __future__ import annotations

import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import gpt_client as gpt

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL may be unavailable in minimal environments.
    Image = None


def log(message: str):
    print(message, flush=True)


def add_common_args(parser):
    parser.add_argument("input_path", type=Path, help="Input JSONL/JSON dataset path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path with appended process supervision.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional JSONL log path.")
    parser.add_argument("--model", default=gpt.DEFAULT_MODEL, help="OpenAI-compatible model used to write traces.")
    parser.add_argument("--api-key", default=None, help="API key. Falls back to env or dataset/openrouter.py.")
    parser.add_argument("--api-base-url", default=gpt.DEFAULT_CHAT_COMPLETIONS_URL, help="OpenAI-compatible chat completions URL.")
    parser.add_argument("--max-tokens", type=int, default=2200, help="Max tokens for each trace-generation call.")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this item index.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of items to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used only for deterministic dry-run stubs.")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent GPT calls.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call GPT; emit deterministic stub traces.")
    parser.add_argument("--ca-bundle", type=Path, default=None, help="Optional CA bundle path for HTTPS requests.")
    parser.add_argument("--insecure-ssl", action="store_true", help="Disable HTTPS certificate verification for GPT calls.")
    parser.add_argument(
        "--messages-output",
        type=Path,
        default=None,
        help="Optional JSONL/JSON output in OpenAI-style multimodal messages format.",
    )
    parser.add_argument(
        "--messages-image-root",
        type=Path,
        default=None,
        help="Optional root path used to relativize image_url paths in --messages-output.",
    )
    parser.add_argument(
        "--messages-image-prefix",
        default="",
        help="Optional prefix prepended after image paths are relativized for --messages-output.",
    )
    parser.add_argument(
        "--messages-doc-loc-prefix",
        default=None,
        help="Optional doc_loc file prefix. Defaults to the ShareGPT output path.",
    )
    parser.add_argument("--messages-lang", default="en", help="Language tag written to --messages-output.")
    return parser


def load_items(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing input dataset: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Expected a JSON array for .json input.")
        return payload
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def init_output_files(output_path: Path, log_path: Path | None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    if output_path.suffix.lower() == ".json":
        output_path.write_text("[]", encoding="utf-8")
    else:
        output_path.write_text("", encoding="utf-8")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()
        log_path.write_text("", encoding="utf-8")


def append_jsonl(path: Path, records):
    if not records:
        return
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_records(output_path: Path, records, snapshot=None):
    if output_path.suffix.lower() == ".json":
        payload = snapshot if snapshot is not None else records
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    append_jsonl(output_path, records)


def read_prompt_text(item) -> str:
    prompt = item.get("prompt") or []
    if not prompt:
        return ""
    content = prompt[0].get("content")
    if isinstance(content, list) and content:
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")
    if isinstance(content, str):
        return content
    return ""


IMG_CONTEXT_INTRO_RE = re.compile(r"\A\s*<IMG_CONTEXT>[\s\S]*?\n\s*\n", re.MULTILINE)
IMAGE_TOKEN_PREFIX_RE = re.compile(r"\A\s*((?:<image>\s*)*)", re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(r"\s*Final answer\s*:\s*([\s\S]+?)\s*$", re.IGNORECASE)


def strip_img_context_intro(prompt_text: str):
    """Remove RL-style image context prose while preserving the actual question."""
    return IMG_CONTEXT_INTRO_RE.sub("", str(prompt_text or ""), count=1).strip()


def normalize_human_value_text(human_value: str):
    """Remove <IMG_CONTEXT> descriptions from an existing LLaMA Factory human value."""
    text = str(human_value or "")
    match = IMAGE_TOKEN_PREFIX_RE.match(text)
    image_tokens = ""
    body = text
    if match:
        image_tokens = "".join(re.findall(r"<image>", match.group(1), flags=re.IGNORECASE))
        body = text[match.end() :]
    body = strip_img_context_intro(body)
    if image_tokens and body:
        return f"{image_tokens}\n{body}"
    if image_tokens:
        return image_tokens
    return body


def extract_image_paths(item):
    extra = item.get("extra_info") or {}
    from_extra = extra.get("prompt_image_paths_abs")
    if isinstance(from_extra, list) and from_extra:
        return [str(Path(path).resolve()) for path in from_extra]
    paths = []
    prompt = item.get("prompt") or []
    if not prompt:
        return paths
    content = prompt[0].get("content")
    if not isinstance(content, list):
        return paths
    for part in content:
        if part.get("type") != "image_url":
            continue
        url = str((part.get("image_url") or {}).get("url") or "")
        if not url:
            continue
        paths.append(url)
    return paths


def convert_prompt_content_for_api(prompt_content):
    converted = []
    for part in prompt_content:
        if part.get("type") != "image_url":
            converted.append(part)
            continue
        payload = dict(part.get("image_url") or {})
        url = str(payload.get("url") or "")
        if url.startswith("data:") or url.startswith("http://") or url.startswith("https://"):
            converted.append({"type": "image_url", "image_url": payload})
            continue
        image_path = Path(url)
        if not image_path.exists():
            continue
        payload["url"] = gpt.encode_image_data_url(image_path)
        converted.append({"type": "image_url", "image_url": payload})
    return converted


def build_api_prompt(item):
    prompt = deepcopy((item.get("prompt") or [])[0])
    if not prompt:
        return {"role": "user", "content": []}
    content = prompt.get("content")
    if isinstance(content, list):
        prompt["content"] = convert_prompt_content_for_api(content)
    return prompt


def gold_answer_payload(item):
    reward_model = item.get("reward_model") or {}
    if "ground_truth" in reward_model:
        ground_truth = reward_model["ground_truth"]
        if isinstance(ground_truth, dict):
            if "value" in ground_truth:
                return {"value": ground_truth["value"]}
            if "values" in ground_truth:
                return {"values": ground_truth["values"]}
        return ground_truth
    return reward_model


def stringify_gold_answer(item):
    payload = gold_answer_payload(item)
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False)


def prompt_answer_candidates(prompt_text: str):
    candidates = []
    for match in re.finditer(r"^-\s*([A-Z])\s*:", str(prompt_text or ""), flags=re.MULTILINE):
        candidates.append(match.group(1))
    allowed_match = re.search(r"Allowed labels:\s*([^\n]+)", str(prompt_text or ""), flags=re.IGNORECASE)
    if allowed_match:
        candidates.extend(part.strip() for part in allowed_match.group(1).split(",") if part.strip())
    relation_match = re.search(r"Allowed relation labels:\s*([^\n]+)", str(prompt_text or ""), flags=re.IGNORECASE)
    if relation_match:
        candidates.extend(part.strip() for part in relation_match.group(1).split(",") if part.strip())
    seen = set()
    deduped = []
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def extract_final_answer_from_text(text: str, prompt_text: str = ""):
    match = FINAL_ANSWER_RE.search(str(text or ""))
    if not match:
        tail = str(text or "").strip()
        tail = re.sub(r"\s+", " ", tail[-800:]).strip()
        sequence_match = re.search(r"([A-Z](?:\s*>\s*[A-Z]){1,})\s*[。.]?$", tail)
        if sequence_match:
            return re.sub(r"\s+", " ", sequence_match.group(1).strip())
        for candidate in sorted(prompt_answer_candidates(prompt_text), key=len, reverse=True):
            if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(candidate)}\s*[。.]?$", tail, flags=re.IGNORECASE):
                return candidate
        label_match = re.search(r"(?:correct\s+label\s+is|answer\s+(?:is|should be)|answer format should be)\s*:?\s*([A-Za-z0-9_.>-]+)\s*[。.]?$", tail, flags=re.IGNORECASE)
        if label_match:
            return label_match.group(1).strip()
        return ""
    return match.group(1).strip()


def remove_final_answer_from_thought(text: str):
    return FINAL_ANSWER_RE.sub("", str(text or "")).strip()


def build_think_answer_value(thought_text: str, final_answer: str):
    thought = remove_final_answer_from_thought(thought_text).strip()
    answer = str(final_answer or "").strip()
    if thought:
        return f"<think>\n{thought}\n</think>\n{answer}"
    return f"<think>\n</think>\n{answer}"


def common_trace_schema():
    return {
        "analysis_summary": "string",
        "reasoning_steps": ["string", "string"],
        "useful_evidence": ["string"],
        "verification_checks": ["string"],
        "final_answer": "string",
        "teacher_response_text": "string",
    }


PLACEHOLDER_PATTERNS = [
    re.compile(r"^\.*$"),
    re.compile(r"^(observation|inference|check|step)\s*:\s*(\.\.\.|…)?\s*$", re.IGNORECASE),
    re.compile(r"^(n/?a|none|unknown|not available|placeholder)\s*$", re.IGNORECASE),
]


def clean_text_fragment(text: str):
    text = str(text or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern.match(text):
            return ""
    return text


def clean_text_list(items):
    cleaned = []
    for item in items or []:
        text = clean_text_fragment(item)
        if text:
            cleaned.append(text)
    return cleaned


def trace_quality_score(trace: dict[str, Any]):
    steps = clean_text_list(trace.get("reasoning_steps") or [])
    checks = clean_text_list(trace.get("verification_checks") or [])
    teacher_text = clean_text_fragment(trace.get("teacher_response_text") or "")
    score = 0
    if len(steps) >= 3:
        score += 2
    elif len(steps) >= 2:
        score += 1
    if checks:
        score += 1
    if teacher_text and len(teacher_text) >= 220:
        score += 2
    elif teacher_text and len(teacher_text) >= 120:
        score += 1
    if "Observation:" in teacher_text and "Final answer:" in teacher_text:
        score += 1
    return score


def render_teacher_response_text(trace: dict[str, Any]):
    teacher_text = clean_text_fragment(trace.get("teacher_response_text") or "")
    if teacher_text:
        return teacher_text
    lines = []
    summary = clean_text_fragment(trace.get("analysis_summary") or "")
    if summary:
        lines.append(f"Overview: {summary}")
    steps = clean_text_list(trace.get("reasoning_steps") or [])
    for idx, step in enumerate(steps, start=1):
        if step:
            lines.append(f"Step {idx}: {step}")
    checks = clean_text_list(trace.get("verification_checks") or [])
    for check in checks:
        if check:
            lines.append(f"Check: {check}")
    final_answer = clean_text_fragment(trace.get("final_answer") or "")
    if final_answer:
        lines.append(f"Final answer: {final_answer}")
    return "\n".join(lines).strip()


def normalize_trace(trace: dict[str, Any], gold_answer_text: str):
    normalized = dict(trace or {})
    if not clean_text_fragment(normalized.get("analysis_summary") or ""):
        normalized["analysis_summary"] = "Follow the visible evidence, align it with the task requirement, and verify the final answer."
    normalized["reasoning_steps"] = clean_text_list(normalized.get("reasoning_steps") if isinstance(normalized.get("reasoning_steps"), list) else [])
    normalized["useful_evidence"] = clean_text_list(normalized.get("useful_evidence") if isinstance(normalized.get("useful_evidence"), list) else [])
    normalized["verification_checks"] = clean_text_list(normalized.get("verification_checks") if isinstance(normalized.get("verification_checks"), list) else [])
    if not clean_text_fragment(normalized.get("final_answer") or ""):
        normalized["final_answer"] = gold_answer_text
    normalized["analysis_summary"] = clean_text_fragment(normalized.get("analysis_summary") or normalized["analysis_summary"])
    normalized["teacher_response_text"] = render_teacher_response_text(normalized)
    return normalized


def build_teacher_messages(task_name: str, item, helpful_info: dict[str, Any], task_instructions: list[str]):
    prompt_message = build_api_prompt(item)
    payload = {
        "task_family": task_name,
        "instructions": task_instructions
        + [
            "You are given the verified gold answer. Do not debate it; derive a faithful reasoning path that arrives at it.",
            "Treat this as teacher-forcing supervision for a strong model: produce a trace that is explicit, grounded, and strategically useful.",
            "Use an Observation -> Inference -> Check style. Each step should either extract evidence, transform evidence into a conclusion, eliminate a nearby alternative, or verify the final format.",
            "Cite the most relevant evidence from the prompt and the provided helpful info instead of speaking in generic terms.",
            "When candidates or distractors exist, explicitly mention why the correct answer fits better than the strongest alternatives.",
            "Use 5 to 10 concise but substantive reasoning steps.",
            "Before the final answer, include a short verification pass that checks answer content and answer format against the task requirements.",
            "Do not leak hidden implementation details, but it is fine to use the helpful info as teacher-side guidance for a faithful trace.",
            "The final_answer field must restate the provided gold answer exactly in text form.",
            "The teacher_response_text should read like a high-quality worked solution: short overview, numbered reasoning steps, a brief verification line, and the final answer.",
            "Every field in return_schema is required, and teacher_response_text must not be empty.",
            "Never use placeholders such as 'Observation: ...', 'Step ...', 'N/A', or empty bullet points. If a kind of evidence is unavailable, omit that line and use another concrete observation instead.",
        ],
        "gold_answer": stringify_gold_answer(item),
        "helpful_info": helpful_info,
        "return_schema": common_trace_schema(),
    }
    teacher_request = {
        "role": "user",
        "content": (
            "Generate the process-supervision trace now.\n"
            "Return ONLY one valid JSON object matching return_schema.\n"
            "Do NOT answer the original task directly.\n"
            "Do NOT return only a label such as A, B, C, D, True, False, or a short final answer.\n"
            "The JSON must include teacher_response_text containing the reasoning process, and final_answer must exactly match gold_answer.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        ),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an expert scientific-document reasoning teacher generating high-quality process supervision "
                "for distilling GPT-5.4-level multimodal reasoning. Produce faithful, evidence-grounded, step-by-step "
                "worked solutions that maximize clarity, verification, and transfer. You must return only valid JSON "
                "for the process-supervision schema, never a bare task answer."
            ),
        },
        prompt_message,
        teacher_request,
    ]


def build_stub_trace(task_name: str, item, helpful_info: dict[str, Any]):
    prompt_text = read_prompt_text(item)
    gold = stringify_gold_answer(item)
    evidence = []
    if helpful_info:
        for key, value in helpful_info.items():
            if value in (None, "", [], {}):
                continue
            evidence.append(f"{key}: {json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value}")
            if len(evidence) >= 3:
                break
    if not evidence and prompt_text:
        evidence.append(prompt_text[:160])
    steps = [
        "Read the task carefully and determine the exact answer format that the final response must follow.",
        "Extract the most relevant visual or textual evidence from the prompt and from the structured hints.",
        "Link that evidence to the verified answer while contrasting it with the most likely alternative interpretations.",
        "Run a short verification pass to make sure the content and output format both match the target answer.",
        "Restate the answer in the exact target format.",
    ]
    response = "\n".join(
        ["Overview: Work backward from the verified answer while grounding each step in visible evidence."]
        + [f"Step {idx + 1}: {step}" for idx, step in enumerate(steps)]
        + ["Check: The answer has been aligned with the required output format and the provided gold answer."]
        + [f"Final answer: {gold}"]
    )
    return {
        "analysis_summary": f"Stub trace for {task_name}.",
        "reasoning_steps": steps,
        "useful_evidence": evidence,
        "verification_checks": [
            "Confirmed that the explanation follows the target answer format.",
            "Confirmed that the final answer text matches the verified gold answer.",
        ],
        "final_answer": gold,
        "teacher_response_text": response,
    }


def build_trace_repair_messages(task_name: str, item, helpful_info: dict[str, Any], task_instructions: list[str], bad_trace: dict[str, Any]):
    prompt_message = build_api_prompt(item)
    payload = {
        "task_family": task_name,
        "repair_goal": "Rewrite the weak trace into a stronger teacher trace with concrete observations, reasoning, verification, and the same verified final answer.",
        "quality_rules": [
            "Do not use placeholders or ellipses.",
            "Every reasoning step must contain concrete evidence or a concrete inference.",
            "Include at least 4 substantive steps unless the task is trivially short.",
            "Mention why the correct answer beats the strongest nearby alternative when applicable.",
            "End with a short verification and the exact final answer text.",
        ],
        "task_specific_instructions": task_instructions,
        "gold_answer": stringify_gold_answer(item),
        "helpful_info": helpful_info,
        "weak_trace_to_rewrite": bad_trace,
        "return_schema": common_trace_schema(),
    }
    teacher_request = {
        "role": "user",
        "content": (
            "Repair the process-supervision trace now.\n"
            "Return ONLY one valid JSON object matching return_schema.\n"
            "Do NOT answer the original task directly.\n"
            "Do NOT return only a label such as A, B, C, D, True, False, or a short final answer.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        ),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are repairing weak multimodal teacher traces. Replace vague or placeholder-heavy explanations "
                "with concrete, evidence-grounded reasoning while preserving the verified final answer. You must return "
                "only valid JSON for the process-supervision schema, never a bare task answer."
            ),
        },
        prompt_message,
        teacher_request,
    ]


def build_json_retry_messages(base_messages, bad_content: str, gold_answer_text: str):
    retry_messages = deepcopy(base_messages)
    retry_messages.append(
        {
            "role": "assistant",
            "content": str(bad_content or "")[:2000],
        }
    )
    retry_messages.append(
        {
            "role": "user",
            "content": (
                "The previous response is invalid because it is not the required JSON trace. "
                "Rewrite it now as exactly one valid JSON object and nothing else. "
                "Do not return a bare answer label. "
                f"The final_answer field must be exactly: {gold_answer_text}\n"
                f"Required schema: {json.dumps(common_trace_schema(), ensure_ascii=False)}"
            ),
        }
    )
    return retry_messages


def call_trace_json(api_key: str, model: str, messages, max_tokens: int, temperature: float, gold_answer_text: str):
    try:
        return gpt.call_json(api_key, model, messages, max_tokens=max_tokens, temperature=temperature)
    except gpt.ModelContentJSONError as exc:
        retry_messages = build_json_retry_messages(messages, exc.content, gold_answer_text)
        return gpt.call_json(api_key, model, retry_messages, max_tokens=max_tokens, temperature=0.1)


def generate_trace(task_name: str, item, helpful_info: dict[str, Any], task_instructions: list[str], args):
    gold_answer_text = stringify_gold_answer(item)
    if args.dry_run:
        return normalize_trace(build_stub_trace(task_name, item, helpful_info), gold_answer_text)
    messages = build_teacher_messages(task_name, item, helpful_info, task_instructions)
    api_key = gpt.load_api_key(args.api_key, Path(__file__).resolve().parent)
    if not api_key:
        raise RuntimeError("API key not found. Pass --api-key or set OPENAI_COMPAT_API_KEY / OPENROUTER_API_KEY.")
    trace = normalize_trace(
        call_trace_json(api_key, args.model, messages, args.max_tokens, 0.35, gold_answer_text),
        gold_answer_text,
    )
    if trace_quality_score(trace) < 4:
        repair_messages = build_trace_repair_messages(task_name, item, helpful_info, task_instructions, trace)
        repaired = normalize_trace(
            call_trace_json(api_key, args.model, repair_messages, args.max_tokens, 0.25, gold_answer_text),
            gold_answer_text,
        )
        if trace_quality_score(repaired) >= trace_quality_score(trace):
            trace = repaired
    return trace


def attach_process_supervision(item, task_name: str, trace: dict[str, Any], model_name: str, dry_run: bool):
    new_item = deepcopy(item)
    normalized_trace = normalize_trace(trace, stringify_gold_answer(item))
    new_item["process_supervision"] = {
        "task_family": task_name,
        "teacher_model": "dry_run_stub" if dry_run else model_name,
        "schema_version": 1,
        "analysis_summary": normalized_trace.get("analysis_summary", ""),
        "reasoning_steps": normalized_trace.get("reasoning_steps", []),
        "useful_evidence": normalized_trace.get("useful_evidence", []),
        "verification_checks": normalized_trace.get("verification_checks", []),
        "final_answer": normalized_trace.get("final_answer", ""),
        "teacher_response_text": normalized_trace.get("teacher_response_text", ""),
    }
    return new_item


def build_llamafactory_record(item, trace: dict[str, Any]):
    normalized_trace = normalize_trace(trace, stringify_gold_answer(item))
    prompt_text = strip_img_context_intro(read_prompt_text(item))
    image_paths = extract_image_paths(item)
    human_prefix = "<image>" * len(image_paths)
    if human_prefix and prompt_text:
        human_value = f"{human_prefix}\n{prompt_text}"
    elif human_prefix:
        human_value = human_prefix
    else:
        human_value = prompt_text
    return {
        "conversations": [
            {
                "from": "human",
                "value": human_value,
            },
            {
                "from": "gpt",
                "value": build_think_answer_value(
                    normalized_trace.get("teacher_response_text", ""),
                    stringify_gold_answer(item),
                ),
            },
        ],
        "images": image_paths,
    }


def extract_prompt_image_payloads(item):
    payloads = []
    prompt = item.get("prompt") or []
    if prompt:
        content = prompt[0].get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    payload = dict(part.get("image_url") or {})
                    if payload.get("url"):
                        payloads.append(payload)
    if payloads:
        return payloads
    return [{"url": path} for path in extract_image_paths(item)]


def image_wh_from_payload(payload):
    wh = payload.get("image_wh")
    if isinstance(wh, (list, tuple)) and len(wh) == 2:
        try:
            return [int(wh[0]), int(wh[1])]
        except Exception:
            return list(wh)
    url = str(payload.get("url") or "")
    if not url or url.startswith(("data:", "http://", "https://")) or Image is None:
        return None
    path = Path(url)
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            width, height = image.size
        return [int(width), int(height)]
    except Exception:
        return None


def relativize_for_messages(url: str, root: Path | None = None, prefix: str = ""):
    text = str(url or "")
    if not text or text.startswith(("data:", "http://", "https://")):
        return text
    path = Path(text)
    rel_text = text
    if root is not None:
        try:
            rel_text = str(path.resolve().relative_to(root.resolve()))
        except Exception:
            rel_text = text
    elif "/Scidocbench/" in text:
        rel_text = text.split("/Scidocbench/", 1)[1]
    if prefix:
        return f"{prefix.rstrip('/')}/{rel_text.lstrip('/')}"
    return rel_text


def doc_loc_path_for_messages(output_path: Path, args):
    if getattr(args, "messages_doc_loc_prefix", None):
        return str(args.messages_doc_loc_prefix)
    text = str(output_path)
    if "/Scidocbench/" in text:
        return "scidocbench/" + text.split("/Scidocbench/", 1)[1].lstrip("/")
    return text


def build_messages_prompt_text(item):
    question_text = strip_img_context_intro(read_prompt_text(item))
    if question_text:
        return f"<IMG_CONTEXT>\n{question_text}"
    return "<IMG_CONTEXT>"


def build_messages_image_parts(item, args):
    parts = []
    root = getattr(args, "messages_image_root", None)
    prefix = getattr(args, "messages_image_prefix", "") or ""
    for payload in extract_prompt_image_payloads(item):
        image_url = {
            "url": relativize_for_messages(str(payload.get("url") or ""), root=root, prefix=prefix),
        }
        wh = image_wh_from_payload(payload)
        if wh is not None:
            image_url["image_wh"] = wh
        parts.append({"type": "image_url", "image_url": image_url})
    return parts


def build_messages_assistant_content(trace: dict[str, Any], item):
    normalized_trace = normalize_trace(trace, stringify_gold_answer(item))
    value = build_think_answer_value(
        normalized_trace.get("teacher_response_text", ""),
        stringify_gold_answer(item),
    )
    return value.replace("</think>\n", "</think>\n\n", 1)


def build_openai_messages_record(item, trace: dict[str, Any], record_id: int, args):
    content = build_messages_image_parts(item, args)
    content.append({"type": "text", "text": build_messages_prompt_text(item)})
    return {
        "id": record_id,
        "messages": [
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "assistant",
                "content": build_messages_assistant_content(trace, item),
            },
        ],
        "doc_loc": f"{doc_loc_path_for_messages(args.output, args)}:{record_id}",
        "lang": getattr(args, "messages_lang", "en") or "en",
    }


def generate_one_record(task_name: str, item, global_index: int, build_helpful_info, task_instructions, args):
    question_id = ((item.get("extra_info") or {}).get("question_id")) or f"item_{global_index:06d}"
    helpful_info = build_helpful_info(item)
    trace = generate_trace(task_name, item, helpful_info, task_instructions, args)
    record = build_llamafactory_record(item, trace)
    return question_id, record, trace


def run_generation_loop(task_name: str, selected, build_helpful_info, task_instructions, args):
    output_records = []
    messages_records = []
    indexed_items = [(args.start_index + idx, item) for idx, item in enumerate(selected)]
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    if getattr(args, "messages_output", None) is not None:
        init_output_files(args.messages_output, None)

    def handle_success(question_id: str, item, record, trace):
        record_id = len(output_records)
        output_records.append(record)
        write_records(args.output, [record], output_records)
        if getattr(args, "messages_output", None) is not None:
            messages_record = build_openai_messages_record(item, trace, record_id, args)
            messages_records.append(messages_record)
            write_records(args.messages_output, [messages_record], messages_records)
        log(f"[item done] {question_id} written={len(output_records)}")

    def handle_failure(global_index: int, item, exc: Exception):
        question_id = ((item.get("extra_info") or {}).get("question_id")) or f"item_{global_index:06d}"
        record = {
            "type": "item_skip",
            "task": task_name,
            "question_id": question_id,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
        if args.log_file is not None:
            append_jsonl(args.log_file, [record])
        log(f"[item skip] {question_id} reason={type(exc).__name__}: {exc}")

    if workers == 1:
        for local_index, (global_index, item) in enumerate(indexed_items, start=1):
            question_id = ((item.get("extra_info") or {}).get("question_id")) or f"item_{global_index:06d}"
            log(f"[item {local_index}/{len(indexed_items)}] start {question_id}")
            try:
                question_id, record, trace = generate_one_record(
                    task_name, item, global_index, build_helpful_info, task_instructions, args
                )
                handle_success(question_id, item, record, trace)
            except Exception as exc:
                handle_failure(global_index, item, exc)
        return output_records

    log(f"[workers] running with {workers} concurrent GPT calls")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_meta = {}
        for local_index, (global_index, item) in enumerate(indexed_items, start=1):
            question_id = ((item.get("extra_info") or {}).get("question_id")) or f"item_{global_index:06d}"
            log(f"[item {local_index}/{len(indexed_items)}] submit {question_id}")
            future = executor.submit(
                generate_one_record,
                task_name,
                item,
                global_index,
                build_helpful_info,
                task_instructions,
                args,
            )
            future_to_meta[future] = (global_index, item)
        for future in as_completed(future_to_meta):
            global_index, item = future_to_meta[future]
            try:
                question_id, record, trace = future.result()
                handle_success(question_id, item, record, trace)
            except Exception as exc:
                handle_failure(global_index, item, exc)
    return output_records
