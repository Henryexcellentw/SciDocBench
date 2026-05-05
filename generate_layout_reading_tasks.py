#!/usr/bin/env python3
"""Generate hard layout-analysis and reading-flow tasks from scientific-doc IR.

Tasks:
1. block_role_classification
2. parent_child_linking
3. reading_order_prediction
4. next_hop_reading_target_prediction

Pipeline:
- Mine high-signal candidates from `document_ir_v2.json`.
- Build multimodal generation payloads (page images + IR summaries).
- Ask GPT-5.4 via OpenRouter to write hard but answerable task stems.
- For heuristic gold labels, run a verifier pass; optionally verify all tasks.
- Emit dataset items in the multimodal JSON format shown by the user.

The script is intentionally standard-library only so it can run in a lean Linux
environment after `texlive/poppler` setup.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import ssl
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


DEFAULT_CHAT_COMPLETIONS_URL = os.environ.get(
    "OPENAI_COMPAT_BASE_URL",
    "http://35.220.164.252:3888/v1/chat/completions",
)
DEFAULT_MODEL = os.environ.get("OPENAI_COMPAT_MODEL", "gpt-5.4")
DATA_SOURCE = "scidoc_layout_reading_a1_v1"
ROLE_LABELS = ["Title", "Abstract", "Text", "Text-EQ", "Figure", "Caption", "Table", "Equation", "Footnote"]
LINK_LABELS = ["caption_target", "continuation", "none"]
TEXT_ORDER_CATEGORIES = {"Abstract", "Text", "Text-EQ"}
MAX_TEXT_PREVIEW = 220
LETTER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
REF_RE = re.compile(r"\b(Figure|Fig\.?|Table|Section|Appendix)\s*([A-Za-z]?\d+)\b", re.I)
LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?")
DISCOURSE_PHASE_SPECS = [
    ("abstract", [r"\babstract\b"]),
    ("introduction", [r"\bintroduction\b", r"\bbackground\b", r"\boverview\b", r"\bmotivation\b", r"\bprelim", r"\bpreliminary\b"]),
    ("related_work", [r"\brelated work\b", r"\bprevious work\b", r"\bliterature\b"]),
    ("data", [r"\bdataset\b", r"\bdata\b", r"\bcorpus\b", r"\bbenchmark\b", r"\bobservation", r"\bobservations\b"]),
    ("method", [r"\bmethods?\b", r"\bapproaches?\b", r"\bmodels?\b", r"\bframeworks?\b", r"\balgorithms?\b", r"\barchitectures?\b"]),
    ("experiment", [r"\bexperiments?\b", r"\bevaluation\b", r"\bsetup\b", r"\bimplementation\b", r"\btraining\b", r"\bsimulations?\b"]),
    ("result", [r"\bresults?\b", r"\banalysis\b", r"\bablation\b", r"\bdiscussion\b", r"\bcomparison\b"]),
    ("conclusion", [r"\bconclusions?\b", r"\bfuture work\b", r"\bsummary\b"]),
]
DISCOURSE_PHASE_RANK = {
    "abstract": 0,
    "introduction": 1,
    "related_work": 2,
    "data": 3,
    "method": 4,
    "experiment": 5,
    "result": 6,
    "conclusion": 7,
}
OPENROUTER_SSL_CONTEXT = None
OPENROUTER_SSL_MODE = "default"
CHAT_COMPLETIONS_URL = DEFAULT_CHAT_COMPLETIONS_URL


def parse_args():
    parser = argparse.ArgumentParser(description="Generate hard layout-reading tasks from document IR.")
    parser.add_argument("doc_dir", type=Path, help="Document directory containing document_ir_v2.json and page images.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Chat-completions model used for generation and verification.")
    parser.add_argument("--api-key", default=None, help="API key. Falls back to env or dataset/openrouter.py.")
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_CHAT_COMPLETIONS_URL,
        help="OpenAI-compatible chat completions URL. Defaults to the shared internal endpoint.",
    )
    parser.add_argument("--role-count", type=int, default=6, help="Number of block-role samples to generate.")
    parser.add_argument("--link-count", type=int, default=6, help="Number of parent-child/linking samples to generate.")
    parser.add_argument("--order-count", type=int, default=6, help="Number of reading-order samples to generate.")
    parser.add_argument("--next-hop-count", type=int, default=6, help="Number of next-hop samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verify-all", action="store_true", help="Also verify direct-gold tasks with a second model pass.")
    parser.add_argument(
        "--verify-reading-order",
        action="store_true",
        help="Run the extra verifier pass for reading-order items. Disabled by default so algorithmic reading-order gold is kept.",
    )
    parser.add_argument(
        "--verify-next-hop",
        action="store_true",
        help="Run the extra verifier pass for next-hop items. Disabled by default so algorithmic/reference gold is kept.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not call the chat-completions API; only emit candidate packs.")
    parser.add_argument("--max-tokens", type=int, default=2500, help="Max tokens for each chat-completions API call.")
    parser.add_argument(
        "--ca-bundle",
        type=Path,
        default=None,
        help="Optional CA bundle path for HTTPS requests. Useful on clusters with custom certificate stores.",
    )
    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable HTTPS certificate verification for OpenRouter requests. Only use this on a trusted network.",
    )
    parser.add_argument(
        "--disable-question-rewrite",
        action="store_true",
        help="Skip the final GPT paraphrase pass that diversifies each finished question.",
    )
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_document(doc_dir: Path):
    ir_path = doc_dir / "document_ir_v2.json"
    if not ir_path.exists():
        raise FileNotFoundError(f"Missing document_ir_v2.json: {ir_path}")
    document = read_json(ir_path)
    return document, ir_path


def load_api_key(cli_value: str | None, script_path: Path):
    if cli_value:
        return cli_value
    for env_name in ["OPENAI_COMPAT_API_KEY", "OPENROUTER_API_KEY"]:
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value
    example_path = script_path.parent / "openrouter.py"
    if example_path.exists():
        content = example_path.read_text(encoding="utf-8")
        for variable_name in ["OPENAI_COMPAT_API_KEY", "OPENROUTER_API_KEY"]:
            match = re.search(rf'{variable_name}\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
    return None


def configure_chat_completions_url(api_base_url: str | None = None):
    global CHAT_COMPLETIONS_URL
    CHAT_COMPLETIONS_URL = (api_base_url or DEFAULT_CHAT_COMPLETIONS_URL).strip()
    return CHAT_COMPLETIONS_URL


def discover_ca_bundle():
    candidates = []
    env_path = os.environ.get("SSL_CERT_FILE")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("/etc/ssl/certs/ca-certificates.crt"),
            Path("/etc/pki/tls/certs/ca-bundle.crt"),
            Path("/etc/ssl/cert.pem"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    try:
        import certifi  # type: ignore

        certifi_path = Path(certifi.where())
        if certifi_path.exists():
            return certifi_path
    except Exception:
        pass
    return None


def configure_openrouter_ssl(ca_bundle: Path | None = None, insecure_ssl: bool = False):
    global OPENROUTER_SSL_CONTEXT, OPENROUTER_SSL_MODE
    if insecure_ssl:
        OPENROUTER_SSL_CONTEXT = ssl._create_unverified_context()
        OPENROUTER_SSL_MODE = "insecure"
        return OPENROUTER_SSL_MODE

    bundle = ca_bundle or discover_ca_bundle()
    if bundle is not None:
        OPENROUTER_SSL_CONTEXT = ssl.create_default_context(cafile=str(bundle))
        OPENROUTER_SSL_MODE = f"ca_bundle:{bundle}"
        return OPENROUTER_SSL_MODE

    OPENROUTER_SSL_CONTEXT = ssl.create_default_context()
    OPENROUTER_SSL_MODE = "system_default"
    return OPENROUTER_SSL_MODE


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\n": " ",
        "\r": " ",
        "\t": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def block_text_preview(block):
    content = block.get("content") or {}
    text = sanitize_text((content.get("plain_text") or "").strip())
    if not text:
        metadata = block.get("metadata") or {}
        text = sanitize_text((metadata.get("caption_text") or "").strip())
    if not text:
        text = sanitize_text((content.get("source_code") or "").strip())
    text = LATEX_COMMAND_RE.sub(" ", text)
    text = sanitize_text(text)
    return text[:MAX_TEXT_PREVIEW]


def bbox_to_list(bbox):
    return [round(bbox["x1"], 1), round(bbox["y1"], 1), round(bbox["x2"], 1), round(bbox["y2"], 1)]


def image_relative_path(doc_dir: Path, image_path: Path):
    return str(image_path.relative_to(doc_dir))


def page_image_path(doc_dir: Path, page_meta):
    image_name = page_meta.get("image_path") or f"page_{page_meta['page_index']:04d}.jpg"
    path = doc_dir / image_name
    if not path.exists():
        raise FileNotFoundError(f"Missing page image: {path}")
    return path


def encode_image_data_url(path: Path):
    suffix = path.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def call_openrouter_json(api_key: str, model: str, messages, max_tokens: int, temperature: float = 0.2, retries: int = 3):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                CHAT_COMPLETIONS_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            scheme = urllib.parse.urlparse(CHAT_COMPLETIONS_URL).scheme.lower()
            if scheme == "https":
                response_ctx = urllib.request.urlopen(request, timeout=180, context=OPENROUTER_SSL_CONTEXT)
            else:
                response_ctx = urllib.request.urlopen(request, timeout=180)
            with response_ctx as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            content = ((response_payload.get("choices") or [{}])[0].get("message") or {}).get("content")
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if not content:
                raise RuntimeError("Chat-completions endpoint returned empty content.")
            return maybe_extract_json(content)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(8, 2 ** attempt))
    if last_error is not None and "CERTIFICATE_VERIFY_FAILED" in str(last_error):
        raise RuntimeError(
            "API SSL verification failed. Try --ca-bundle /etc/ssl/certs/ca-certificates.crt "
            "or --insecure-ssl if you trust the network. "
            f"Current ssl_mode={OPENROUTER_SSL_MODE}. Original error: {last_error}"
        ) from last_error
    raise last_error


def maybe_extract_json(text: str):
    if not text:
        raise RuntimeError("Expected JSON, got empty text.")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def root_block_id(block_id, block_by_id):
    current = block_id
    seen = set()
    while current in block_by_id and current not in seen:
        seen.add(current)
        parent = (block_by_id[current].get("structure") or {}).get("parent_block")
        if parent is None or parent not in block_by_id:
            return current
        current = parent
    return current


def page_sort_key(block):
    bbox = block["bbox"]
    return (block["page_index"], bbox["y1"], bbox["x1"], bbox["x2"])


def build_doc_context(document):
    pages_by_index = {page["page_index"]: page for page in document.get("pages", [])}
    block_by_id = {block["block_id"]: block for block in document.get("blocks", [])}
    blocks_by_page = defaultdict(list)
    for block in document.get("blocks", []):
        blocks_by_page[block["page_index"]].append(block)
    for page_index in blocks_by_page:
        blocks_by_page[page_index].sort(key=page_sort_key)

    # Enumerate figures/tables by order of caption anchors.
    figure_anchors = []
    table_anchors = []
    for block in sorted(document.get("blocks", []), key=page_sort_key):
        target_ids = (block.get("metadata") or {}).get("caption_target_block_ids") or []
        target_type = (block.get("metadata") or {}).get("caption_target_type")
        if block.get("category_name") == "Caption" and target_ids and target_type == "Figure":
            figure_anchors.append((block, target_ids[0]))
        if block.get("category_name") == "Caption" and target_ids and target_type == "Table":
            table_anchors.append((block, target_ids[0]))

    figure_number_to_caption = {}
    figure_number_to_figure = {}
    for idx, (caption_block, target_id) in enumerate(figure_anchors, start=1):
        figure_number_to_caption[str(idx)] = caption_block["block_id"]
        figure_number_to_figure[str(idx)] = target_id

    table_number_to_caption = {}
    table_number_to_table = {}
    for idx, (caption_block, target_id) in enumerate(table_anchors, start=1):
        table_number_to_caption[str(idx)] = caption_block["block_id"]
        table_number_to_table[str(idx)] = target_id

    # Enumerate unique title roots as section anchors.
    title_roots = []
    seen_roots = set()
    for block in sorted(document.get("blocks", []), key=page_sort_key):
        if block.get("category_name") != "Title":
            continue
        rid = root_block_id(block["block_id"], block_by_id)
        if rid in seen_roots:
            continue
        seen_roots.add(rid)
        title_roots.append(block_by_id[rid])

    section_number_to_title = {}
    for idx, block in enumerate(sorted(title_roots, key=page_sort_key), start=1):
        section_number_to_title[str(idx)] = block["block_id"]

    return {
        "pages_by_index": pages_by_index,
        "block_by_id": block_by_id,
        "blocks_by_page": blocks_by_page,
        "figure_number_to_caption": figure_number_to_caption,
        "figure_number_to_figure": figure_number_to_figure,
        "table_number_to_caption": table_number_to_caption,
        "table_number_to_table": table_number_to_table,
        "section_number_to_title": section_number_to_title,
        "title_roots_in_order": [block["block_id"] for block in sorted(title_roots, key=page_sort_key)],
    }


def document_order_key(block, ctx):
    page_width = ctx["pages_by_index"][block["page_index"]]["width"]
    return (block["page_index"],) + reading_order_key(block, page_width)


def normalize_label_text(text: str):
    text = sanitize_text(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def infer_discourse_phase(label_text: str):
    normalized = normalize_label_text(label_text)
    for phase, patterns in DISCOURSE_PHASE_SPECS:
        for pattern in patterns:
            if re.search(pattern, normalized):
                return phase
    return None


def build_text_block_context(ctx):
    block_by_id = ctx["block_by_id"]
    ordered_blocks = sorted(block_by_id.values(), key=lambda block: document_order_key(block, ctx))
    block_order_index = {}
    block_to_section_title = {}
    block_to_phase = {}
    current_title_root = None

    for index, block in enumerate(ordered_blocks):
        block_order_index[block["block_id"]] = index
        if block["category_name"] == "Title":
            current_title_root = root_block_id(block["block_id"], block_by_id)
            continue
        if block["category_name"] not in TEXT_ORDER_CATEGORIES:
            continue
        if block["category_name"] == "Abstract":
            block_to_phase[block["block_id"]] = "abstract"
            block_to_section_title[block["block_id"]] = None
            continue
        block_to_section_title[block["block_id"]] = current_title_root
        if current_title_root is not None:
            title_block = block_by_id[current_title_root]
            block_to_phase[block["block_id"]] = infer_discourse_phase(block_text_preview(title_block))
        else:
            block_to_phase[block["block_id"]] = None

    ctx["ordered_block_ids"] = [block["block_id"] for block in ordered_blocks]
    ctx["block_order_index"] = block_order_index
    ctx["block_to_section_title"] = block_to_section_title
    ctx["block_to_phase"] = block_to_phase


def coarse_column(block, page_width):
    bbox = block["bbox"]
    center = (bbox["x1"] + bbox["x2"]) / 2.0
    width_ratio = bbox["width"] / max(1.0, page_width)
    if width_ratio >= 0.68:
        return "full"
    return "left" if center < page_width / 2.0 else "right"


def reading_order_key(block, page_width):
    bbox = block["bbox"]
    category = block["category_name"]
    column = coarse_column(block, page_width)
    if category == "Footnote":
        bucket = 5
    elif column == "full" and bbox["y1"] < 0.35 * page_width:
        bucket = 0
    elif column == "full":
        bucket = 3
    elif column == "left":
        bucket = 1
    else:
        bucket = 2
    cat_priority = 0 if category in {"Title", "Abstract"} else 1
    return (bucket, bbox["y1"], cat_priority, bbox["x1"])


def sample_letters(count):
    return list(LETTER_LABELS[:count])


def relation_pairs(document, ctx):
    block_by_id = ctx["block_by_id"]
    positives = []
    positive_pair_keys = set()

    for block in document.get("blocks", []):
        target_ids = (block.get("metadata") or {}).get("caption_target_block_ids") or []
        if block.get("category_name") == "Caption" and target_ids:
            pair = (block["block_id"], target_ids[0], "caption_target")
            positives.append(pair)
            positive_pair_keys.add((block["block_id"], target_ids[0]))
            positive_pair_keys.add((target_ids[0], block["block_id"]))

    for block in document.get("blocks", []):
        parent = (block.get("structure") or {}).get("parent_block")
        if parent is None or parent not in block_by_id:
            continue
        pair = (parent, block["block_id"], "continuation")
        positives.append(pair)
        positive_pair_keys.add((parent, block["block_id"]))
        positive_pair_keys.add((block["block_id"], parent))

    negatives = []
    rng = random.Random(0)
    for block in document.get("blocks", []):
        same_page = [b for b in ctx["blocks_by_page"][block["page_index"]] if b["block_id"] != block["block_id"]]
        rng.shuffle(same_page)
        for other in same_page:
            if (block["block_id"], other["block_id"]) in positive_pair_keys:
                continue
            same_cat = block["category_name"] == other["category_name"]
            near_x = abs(block["bbox"]["x1"] - other["bbox"]["x1"]) < 250
            near_y = abs(block["bbox"]["y1"] - other["bbox"]["y1"]) < 500
            if same_cat or (near_x and near_y):
                negatives.append((block["block_id"], other["block_id"], "none"))
                break
    return positives, negatives


def score_role_candidate(block, page_block_count):
    score = 0.0
    cat = block["category_name"]
    preview = block_text_preview(block)
    if cat in {"Caption", "Footnote", "Text-EQ", "Title"}:
        score += 3.0
    if len(preview) < 80:
        score += 1.0
    if page_block_count >= 12:
        score += 1.0
    if block["bbox"]["width"] < 350 or block["bbox"]["height"] < 80:
        score += 1.0
    return score


def extract_role_candidates(document, ctx, limit: int):
    scored = []
    for block in document.get("blocks", []):
        if block["category_name"] not in ROLE_LABELS:
            continue
        page_count = len(ctx["blocks_by_page"][block["page_index"]])
        scored.append((score_role_candidate(block, page_count), block))
    scored.sort(key=lambda item: (-item[0], page_sort_key(item[1])))
    results = []
    for _, block in scored[:limit]:
        results.append(
            {
                "task": "block_role_classification",
                "gold_source": "direct",
                "gold_answer": block["category_name"],
                "target_block_id": block["block_id"],
                "page_indices": [block["page_index"]],
                "support_block_ids": [block["block_id"]],
            }
        )
    return results


def extract_linking_candidates(document, ctx, limit: int):
    positives, negatives = relation_pairs(document, ctx)
    candidates = []
    for a_id, b_id, relation in positives + negatives:
        a = ctx["block_by_id"][a_id]
        b = ctx["block_by_id"][b_id]
        difficulty = 2.0
        if relation == "none":
            difficulty += 1.5
        if a["page_index"] != b["page_index"]:
            difficulty += 1.5
        if a["category_name"] == b["category_name"]:
            difficulty += 1.0
        candidates.append(
            (
                difficulty,
                {
                    "task": "parent_child_linking",
                    "gold_source": "direct",
                    "gold_answer": relation,
                    "block_pair": [a_id, b_id],
                    "page_indices": sorted({a["page_index"], b["page_index"]}),
                    "support_block_ids": [a_id, b_id],
                },
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1]["page_indices"], item[1]["block_pair"]))
    return [item[1] for item in candidates[:limit]]


def eligible_text_order_blocks(ctx):
    blocks = []
    for block in ctx["block_by_id"].values():
        if block["category_name"] not in TEXT_ORDER_CATEGORIES:
            continue
        preview = block_text_preview(block)
        if len(preview) < 60:
            continue
        blocks.append(block)
    return sorted(blocks, key=lambda block: ctx["block_order_index"][block["block_id"]])


def extract_local_text_order_candidates(ctx):
    ordered = eligible_text_order_blocks(ctx)
    candidates = []
    for span_size in [7, 6, 5]:
        if len(ordered) < span_size:
            continue
        for start in range(0, len(ordered) - span_size + 1):
            span = ordered[start : start + span_size]
            page_span = span[-1]["page_index"] - span[0]["page_index"]
            if page_span > 1:
                continue
            section_ids = {ctx["block_to_section_title"].get(block["block_id"]) for block in span if block["category_name"] != "Abstract"}
            if len(section_ids - {None}) > 1:
                continue
            pick_count = 4 if span_size >= 6 else 3
            if pick_count == 4:
                positions = [0, 2, span_size - 2, span_size - 1]
            else:
                positions = [0, span_size // 2, span_size - 1]
            selected = [span[idx] for idx in positions]
            block_ids = [block["block_id"] for block in selected]
            previews = [block_text_preview(block) for block in selected]
            if len(set(previews)) != len(previews):
                continue
            difficulty = 5.0 + 0.4 * page_span + 0.3 * pick_count
            if len(section_ids - {None}) == 1:
                difficulty += 1.0
            candidates.append(
                (
                    difficulty,
                    {
                        "task": "reading_order_prediction",
                        "subtype": "local_text_flow",
                        "gold_source": "heuristic_text_flow",
                        "gold_answer": block_ids,
                        "candidate_block_ids": block_ids,
                        "page_indices": sorted({block["page_index"] for block in selected}),
                        "support_block_ids": block_ids,
                    },
                )
            )
    return candidates


def extract_discourse_progression_candidates(ctx):
    by_phase = defaultdict(list)
    for block in eligible_text_order_blocks(ctx):
        phase = ctx["block_to_phase"].get(block["block_id"])
        if phase is None:
            continue
        by_phase[phase].append(block)

    available_phases = sorted(
        by_phase,
        key=lambda phase: min(ctx["block_order_index"][block["block_id"]] for block in by_phase[phase]),
    )
    candidates = []
    for size in [4, 3]:
        if len(available_phases) < size:
            continue
        for start in range(0, len(available_phases) - size + 1):
            phases = available_phases[start : start + size]
            selected = []
            last_index = -1
            for phase in phases:
                block = next(
                    (
                        candidate
                        for candidate in by_phase[phase]
                        if ctx["block_order_index"][candidate["block_id"]] > last_index
                    ),
                    None,
                )
                if block is None:
                    selected = []
                    break
                selected.append(block)
                last_index = ctx["block_order_index"][block["block_id"]]
            if not selected:
                continue
            ranks = [DISCOURSE_PHASE_RANK.get(phase, 99) for phase in phases]
            block_ids = [block["block_id"] for block in selected]
            difficulty = 6.5 + 0.8 * size + 0.3 * len(set(phases))
            if ranks != sorted(ranks):
                difficulty += 0.8
            candidates.append(
                (
                    difficulty,
                    {
                        "task": "reading_order_prediction",
                        "subtype": "discourse_progression_order",
                        "gold_source": "heuristic_discourse",
                        "gold_answer": block_ids,
                        "candidate_block_ids": block_ids,
                        "page_indices": sorted({block["page_index"] for block in selected}),
                        "support_block_ids": block_ids,
                    },
                )
            )
    return candidates


def extract_reading_order_candidates(document, ctx, limit: int):
    candidates = extract_local_text_order_candidates(ctx) + extract_discourse_progression_candidates(ctx)
    seen = set()
    unique = []
    for score, candidate in sorted(candidates, key=lambda item: (-item[0], item[1]["page_indices"], item[1]["candidate_block_ids"])):
        key = (candidate.get("subtype"), tuple(candidate["candidate_block_ids"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique


def extract_explicit_refs(block):
    text = "\n".join(
        [
            (block.get("content") or {}).get("source_code") or "",
            (block.get("content") or {}).get("plain_text") or "",
        ]
    )
    refs = []
    for match in REF_RE.finditer(text):
        ref_type = match.group(1).lower()
        ref_num = match.group(2)
        refs.append({"type": ref_type, "number": ref_num, "text": match.group(0)})
    return refs


def build_reference_target(ref_item, ctx):
    number = re.sub(r"^[A-Za-z]", "", ref_item["number"])
    ref_type = ref_item["type"]
    if ref_type.startswith("fig"):
        caption_id = ctx["figure_number_to_caption"].get(number)
        if caption_id is not None:
            return caption_id, "caption_anchor"
    if ref_type == "table":
        caption_id = ctx["table_number_to_caption"].get(number)
        if caption_id is not None:
            return caption_id, "caption_anchor"
    if ref_type == "section":
        title_id = ctx["section_number_to_title"].get(number)
        if title_id is not None:
            return title_id, "section_anchor"
    return None, None


def extract_next_hop_candidates(document, ctx, limit: int):
    candidates = []
    # Direct caption <-> figure/table hops.
    for block in document.get("blocks", []):
        linked = (block.get("metadata") or {}).get("linked_caption_block_ids") or []
        if block["category_name"] in {"Figure", "Table"} and linked:
            target_id = linked[0]
            difficulty = 3.0
            candidates.append(
                (
                    difficulty,
                    {
                        "task": "next_hop_reading_target_prediction",
                        "subtype": "visual_to_caption",
                        "gold_source": "direct",
                        "gold_answer": target_id,
                        "current_block_id": block["block_id"],
                        "candidate_target_block_ids": None,
                        "page_indices": sorted({block["page_index"], ctx["block_by_id"][target_id]["page_index"]}),
                        "support_block_ids": [block["block_id"], target_id],
                    },
                )
            )
        target_ids = (block.get("metadata") or {}).get("caption_target_block_ids") or []
        if block["category_name"] == "Caption" and target_ids:
            target_id = target_ids[0]
            difficulty = 3.0
            candidates.append(
                (
                    difficulty,
                    {
                        "task": "next_hop_reading_target_prediction",
                        "subtype": "caption_to_visual",
                        "gold_source": "direct",
                        "gold_answer": target_id,
                        "current_block_id": block["block_id"],
                        "candidate_target_block_ids": None,
                        "page_indices": sorted({block["page_index"], ctx["block_by_id"][target_id]["page_index"]}),
                        "support_block_ids": [block["block_id"], target_id],
                    },
                )
            )

    # Reference-following hops from prose.
    for block in document.get("blocks", []):
        if block["category_name"] not in {"Text", "Text-EQ", "Abstract"}:
            continue
        refs = extract_explicit_refs(block)
        for ref_item in refs:
            target_id, subtype = build_reference_target(ref_item, ctx)
            if target_id is None:
                continue
            difficulty = 5.0
            candidates.append(
                (
                    difficulty,
                    {
                        "task": "next_hop_reading_target_prediction",
                        "subtype": subtype,
                        "gold_source": "heuristic_reference",
                        "gold_answer": target_id,
                        "current_block_id": block["block_id"],
                        "reference_text": ref_item["text"],
                        "candidate_target_block_ids": None,
                        "page_indices": sorted({block["page_index"], ctx["block_by_id"][target_id]["page_index"]}),
                        "support_block_ids": [block["block_id"], target_id],
                    },
                )
            )

    candidates.sort(key=lambda item: (-item[0], item[1]["page_indices"]))
    unique = []
    seen = set()
    for _, candidate in candidates:
        key = (candidate["subtype"], candidate.get("current_block_id"), candidate["gold_answer"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique


def choose_distractors(block_ids, gold_id, ctx, limit, preferred_pages=None, exclude_block_ids=None):
    rng = random.Random(gold_id + limit)
    block_by_id = ctx["block_by_id"]
    gold_block = block_by_id[gold_id]
    pool = list(block_by_id.values())
    rng.shuffle(pool)
    selected = []
    used = {gold_id} | set(exclude_block_ids or [])
    for block in pool:
        if block["block_id"] in used:
            continue
        if preferred_pages is not None and block["page_index"] not in preferred_pages and len(selected) < max(1, limit - 1):
            continue
        same_cat = block["category_name"] == gold_block["category_name"]
        near = abs(block["bbox"]["x1"] - gold_block["bbox"]["x1"]) < 350 or abs(block["page_index"] - gold_block["page_index"]) <= 1
        if same_cat or near:
            selected.append(block["block_id"])
            used.add(block["block_id"])
        if len(selected) >= limit:
            break
    return selected


def materialize_candidate(candidate, ctx):
    block_by_id = ctx["block_by_id"]
    task = candidate["task"]

    if task == "block_role_classification":
        target = block_by_id[candidate["target_block_id"]]
        candidate["choice_labels"] = ROLE_LABELS
        candidate["final_answer"] = candidate["gold_answer"]
        candidate["blocks"] = [target["block_id"]]
        return candidate

    if task == "parent_child_linking":
        a_id, b_id = candidate["block_pair"]
        candidate["choice_labels"] = LINK_LABELS
        candidate["final_answer"] = candidate["gold_answer"]
        candidate["blocks"] = [a_id, b_id]
        return candidate

    if task == "reading_order_prediction":
        gold_block_ids = list(candidate["gold_answer"])
        option_block_ids = list(candidate["candidate_block_ids"])
        rng = random.Random(sum(gold_block_ids) + len(option_block_ids) * 17)
        rng.shuffle(option_block_ids)
        if len(option_block_ids) > 1 and option_block_ids == gold_block_ids:
            option_block_ids[0], option_block_ids[1] = option_block_ids[1], option_block_ids[0]
        label_map = {block_id: LETTER_LABELS[idx] for idx, block_id in enumerate(option_block_ids)}
        candidate["label_map"] = label_map
        candidate["candidate_block_ids"] = option_block_ids
        candidate["final_answer"] = " > ".join(label_map[block_id] for block_id in gold_block_ids)
        candidate["blocks"] = option_block_ids
        return candidate

    if task == "next_hop_reading_target_prediction":
        current_block_id = candidate["current_block_id"]
        gold_id = candidate["gold_answer"]
        distractors = choose_distractors(
            block_ids=[],
            gold_id=gold_id,
            ctx=ctx,
            limit=3,
            preferred_pages=candidate["page_indices"],
            exclude_block_ids=[current_block_id],
        )
        option_block_ids = [gold_id] + distractors
        # Keep option order non-trivial but deterministic.
        rng = random.Random(current_block_id + gold_id)
        rng.shuffle(option_block_ids)
        label_map = {block_id: LETTER_LABELS[idx] for idx, block_id in enumerate(option_block_ids)}
        candidate["candidate_target_block_ids"] = option_block_ids
        candidate["label_map"] = label_map
        candidate["final_answer"] = label_map[gold_id]
        candidate["blocks"] = [current_block_id] + option_block_ids
        return candidate

    raise ValueError(f"Unknown task: {task}")


def block_descriptor(block):
    return {
        "block_id": block["block_id"],
        "page_index": block["page_index"],
        "bbox": bbox_to_list(block["bbox"]),
        "preview": block_text_preview(block),
        "category_name": block["category_name"],
        "title_level": (block.get("metadata") or {}).get("title_level"),
        "caption_target_block_ids": (block.get("metadata") or {}).get("caption_target_block_ids") or [],
        "linked_caption_block_ids": (block.get("metadata") or {}).get("linked_caption_block_ids") or [],
        "parent_block": (block.get("structure") or {}).get("parent_block"),
    }


def build_task_facts(candidate, ctx):
    block_by_id = ctx["block_by_id"]
    task = candidate["task"]
    facts = {
        "task": task,
        "gold_source": candidate["gold_source"],
        "page_indices": candidate["page_indices"],
        "blocks": [block_descriptor(block_by_id[block_id]) for block_id in candidate["blocks"]],
    }

    if task == "block_role_classification":
        facts.update(
            {
                "target_block_id": candidate["target_block_id"],
                "gold_label": candidate["gold_answer"],
                "allowed_labels": candidate["choice_labels"],
            }
        )
    elif task == "parent_child_linking":
        facts.update(
            {
                "block_a_id": candidate["block_pair"][0],
                "block_b_id": candidate["block_pair"][1],
                "gold_relation": candidate["gold_answer"],
                "allowed_relations": candidate["choice_labels"],
            }
        )
    elif task == "reading_order_prediction":
        facts.update(
            {
                "subtype": candidate.get("subtype"),
                "candidate_label_to_block_id": {label: block_id for block_id, label in candidate["label_map"].items()},
                "gold_sequence": candidate["final_answer"],
                "candidate_discourse_phase": {
                    label: ctx["block_to_phase"].get(block_id) for block_id, label in candidate["label_map"].items()
                },
                "candidate_section_title_ids": {
                    label: ctx["block_to_section_title"].get(block_id) for block_id, label in candidate["label_map"].items()
                },
            }
        )
    elif task == "next_hop_reading_target_prediction":
        facts.update(
            {
                "current_block_id": candidate["current_block_id"],
                "subtype": candidate.get("subtype"),
                "reference_text": candidate.get("reference_text"),
                "candidate_label_to_block_id": {label: block_id for block_id, label in candidate["label_map"].items()},
                "gold_label": candidate["final_answer"],
                "gold_target_block_id": candidate["gold_answer"],
            }
        )
    return facts


def build_generation_messages(candidate, ctx, doc_dir: Path):
    block_by_id = ctx["block_by_id"]
    facts = build_task_facts(candidate, ctx)
    pages = []
    for page_index in candidate["page_indices"]:
        page_meta = ctx["pages_by_index"][page_index]
        image_path = page_image_path(doc_dir, page_meta)
        pages.append(
            {
                "page_index": page_index,
                "image_path": image_path,
                "image_wh": [page_meta["width"], page_meta["height"]],
            }
        )

    task_specific = {
        "block_role_classification": (
            "Design a very hard scientific-document layout question that asks for the semantic role of exactly one target block. "
            "The final task should give the solver only the page image, the target bbox, and the allowed labels. "
            "Do not refer to hidden preview text or any hidden block content."
        ),
        "parent_child_linking": (
            "Design a very hard relation-classification question about whether two blocks form a direct structural relation. "
            "Allowed answers are exactly the provided relation labels."
        ),
        "reading_order_prediction": (
            "Design a very hard multimodal ordering question. "
            "The final task should show the relevant page image(s), but each candidate block should expose only one clue type: either bbox or preview text, not both. "
            "The solver should recover the correct order from discourse flow, local reading flow, and visual page evidence."
        ),
        "next_hop_reading_target_prediction": (
            "Design a very hard next-hop reading target question. The solver should decide which candidate block to jump to next after the current block."
        ),
    }[candidate["task"]]

    system_prompt = (
        "You are constructing adversarial multimodal evaluation items for scientific document parsing.\n"
        "Use the page images plus the IR-derived block summaries to write a hard but unambiguous task.\n"
        "Do not invent document facts. Keep the final stem concise.\n"
        "Return JSON only with keys: question_stem, answer_format, difficulty_rationale."
    )
    user_text = {
        "instruction": task_specific,
        "facts_for_question_design_only": facts,
        "important_constraints": [
            "The final question must be answerable from the page image(s) and the provided candidate block metadata.",
            "Do not mention hidden gold labels or hidden relation names that are not part of the answer space.",
            "Maximize challenge by emphasizing visually confusing alternatives or reading-flow traps.",
            "The final answer must remain exactly the provided gold answer after validation.",
        ],
    }

    content = [
        {
            "type": "text",
            "text": json.dumps(user_text, ensure_ascii=False, indent=2),
        }
    ]
    for page in pages:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image_data_url(page["image_path"]),
                },
            }
        )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


def build_final_prompt_text(candidate, generated, ctx):
    block_by_id = ctx["block_by_id"]
    lines = []
    for idx, page_index in enumerate(candidate["page_indices"], start=1):
        lines.append("<IMG_CONTEXT>")
        lines.append(f"Image {idx} shows page {page_index} of the same scientific document.")
    lines.append("")
    lines.append(generated["question_stem"].strip())
    lines.append("")

    task = candidate["task"]
    if task == "block_role_classification":
        block = block_by_id[candidate["target_block_id"]]
        lines.append(f"Target block: page {block['page_index']}, bbox={bbox_to_list(block['bbox'])}")
        lines.append(f"Allowed labels: {', '.join(candidate['choice_labels'])}")
    elif task == "parent_child_linking":
        a = block_by_id[candidate["block_pair"][0]]
        b = block_by_id[candidate["block_pair"][1]]
        modes = parent_child_display_modes(candidate)
        lines.append(format_block_single_clue(a, "A", modes[a["block_id"]]))
        lines.append(format_block_single_clue(b, "B", modes[b["block_id"]]))
        lines.append(f"Allowed relation labels: {', '.join(candidate['choice_labels'])}")
    elif task == "reading_order_prediction":
        modes = reading_order_display_modes(candidate)
        lines.append("Candidates:")
        inverse = {label: block_id for block_id, label in candidate["label_map"].items()}
        for label in sorted(inverse):
            block = block_by_id[inverse[label]]
            lines.append(format_block_single_clue(block, label, modes[block["block_id"]]))
    elif task == "next_hop_reading_target_prediction":
        current = block_by_id[candidate["current_block_id"]]
        modes = next_hop_display_modes(candidate)
        lines.append("Current block:")
        lines.append(format_block_single_clue(current, "Current", modes[current["block_id"]]))
        lines.append("Candidates:")
        inverse = {label: block_id for block_id, label in candidate["label_map"].items()}
        for label in sorted(inverse):
            block = block_by_id[inverse[label]]
            lines.append(format_block_single_clue(block, label, modes[block["block_id"]]))

    if generated.get("answer_format"):
        lines.append("")
        lines.append(f"Answer format: {generated['answer_format'].strip()}")
    return "\n".join(lines).strip()


def canonical_question_stem(candidate):
    task = candidate["task"]
    if task == "block_role_classification":
        return "Identify the semantic role of the target block using the page image."
    if task == "parent_child_linking":
        return "Determine the structural relation between block A and block B."
    if task == "reading_order_prediction":
        if candidate.get("subtype") == "discourse_progression_order":
            return "Order the candidate blocks according to the most plausible discourse progression of the paper."
        return "Order the candidate blocks according to the most plausible local reading flow."
    if task == "next_hop_reading_target_prediction":
        return "Choose the most plausible next block to read after the current block."
    raise ValueError(f"Unknown task: {task}")


def canonical_answer_format(candidate):
    task = candidate["task"]
    if task == "block_role_classification":
        return "Return exactly one label from the allowed list."
    if task == "parent_child_linking":
        return "Return exactly one label: caption_target, continuation, or none."
    if task == "reading_order_prediction":
        return "Return one sequence over the candidate labels, for example: A > B > C."
    if task == "next_hop_reading_target_prediction":
        return "Return only one candidate label."
    raise ValueError(f"Unknown task: {task}")


def build_question_rewrite_messages(canonical_question_stem_text: str):
    system_prompt = (
        "You are rewriting a single benchmark question sentence.\n"
        "Make only a light paraphrase that preserves the exact meaning.\n"
        "Do not add any hints, examples, rationale, block ids, page numbers, spatial descriptions, quoted text, or any new information.\n"
        "Keep the rewritten sentence similar in length to the original, or shorter.\n"
        "Return JSON only with keys: rewritten_question_stem, rewrite_notes."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": canonical_question_stem_text},
    ]


def rewrite_question_stem(canonical_question_stem_text: str, api_key: str, model: str, max_tokens: int):
    messages = build_question_rewrite_messages(canonical_question_stem_text)
    rewritten = call_openrouter_json(api_key, model, messages, max_tokens=min(max_tokens, 600), temperature=0.6)
    rewritten_question_stem = sanitize_text_block(rewritten.get("rewritten_question_stem", ""))
    if not rewritten_question_stem:
        raise RuntimeError("Question rewrite returned empty rewritten_question_stem.")
    if "\n" in rewritten_question_stem:
        raise RuntimeError("Question rewrite returned a multi-line stem.")
    return rewritten_question_stem, rewritten


def sanitize_text_block(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def format_block_single_clue(block, label: str, mode: str):
    if mode == "bbox":
        return f"- {label}: page {block['page_index']}, bbox={bbox_to_list(block['bbox'])}"
    if mode == "preview":
        return f"- {label}: preview=\"{block_text_preview(block)}\""
    raise ValueError(f"Unknown block display mode: {mode}")


def parent_child_display_modes(candidate):
    a_id, b_id = candidate["block_pair"]
    if (a_id + b_id) % 2 == 0:
        return {a_id: "bbox", b_id: "preview"}
    return {a_id: "preview", b_id: "bbox"}


def next_hop_display_modes(candidate):
    current_id = candidate["current_block_id"]
    option_ids = list(candidate["candidate_target_block_ids"] or [])
    modes = {current_id: "preview"}
    if not option_ids:
        return modes
    for block_id in option_ids:
        modes[block_id] = "bbox" if (current_id + block_id) % 2 == 0 else "preview"
    option_modes = [modes[block_id] for block_id in option_ids]
    if len(option_ids) > 1 and len(set(option_modes)) == 1:
        first = option_ids[0]
        modes[first] = "preview" if modes[first] == "bbox" else "bbox"
    return modes


def reading_order_display_modes(candidate):
    block_ids = list(candidate["candidate_block_ids"])
    modes = {}
    for idx, block_id in enumerate(block_ids):
        modes[block_id] = "bbox" if idx % 2 == 0 else "preview"
    if len(block_ids) > 1 and len(set(modes.values())) == 1:
        first = block_ids[0]
        modes[first] = "preview" if modes[first] == "bbox" else "bbox"
    return modes


def protected_prompt_lines(prompt_text: str):
    protected = []
    for line in prompt_text.split("\n"):
        stripped = line.strip()
        if (
            "<IMG_CONTEXT>" in stripped
            or "bbox=" in stripped
            or "preview=" in stripped
            or stripped.startswith("- ")
            or stripped.startswith("Allowed labels:")
            or stripped.startswith("Allowed relation labels:")
            or stripped.startswith("Answer format:")
        ):
            protected.append(stripped)
    return protected


def validate_rewritten_prompt_text(canonical_prompt_text: str, rewritten_prompt_text: str):
    canonical_protected = protected_prompt_lines(canonical_prompt_text)
    rewritten_protected = protected_prompt_lines(rewritten_prompt_text)
    missing = [line for line in canonical_protected if line not in rewritten_protected]
    if missing:
        raise RuntimeError(f"Question rewrite changed protected prompt lines: {missing[:3]}")


def build_final_dataset_item(candidate, generated, ctx, doc_dir: Path, prompt_text: str | None = None):
    prompt_text = prompt_text or build_final_prompt_text(candidate, generated, ctx)
    prompt_content = [{"type": "text", "text": prompt_text}]
    for page_index in candidate["page_indices"]:
        page_meta = ctx["pages_by_index"][page_index]
        img_path = page_image_path(doc_dir, page_meta)
        prompt_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_relative_path(doc_dir, img_path),
                    "image_wh": [page_meta["width"], page_meta["height"]],
                },
            }
        )

    item = {
        "prompt": [{"role": "user", "content": prompt_content}],
        "reward_model": {"ground_truth": candidate["final_answer"]},
        "data_source": DATA_SOURCE,
        "extra_info": {
            "task": candidate["task"],
            "gold_source_type": candidate["gold_source"],
            "doc_id": doc_dir.name,
            "page_indices": candidate["page_indices"],
            "support_block_ids": candidate["support_block_ids"],
            "canonical_question_stem": generated.get("canonical_question_stem"),
            "final_question_stem": generated["question_stem"],
        },
    }
    if generated.get("rewrite_notes") is not None:
        item["extra_info"]["rewrite_notes"] = generated.get("rewrite_notes")
    if generated.get("canonical_prompt_text") is not None:
        item["extra_info"]["canonical_prompt_text"] = generated.get("canonical_prompt_text")
    item["extra_info"]["question_rewritten"] = bool(generated.get("question_rewritten"))
    if candidate["task"] == "next_hop_reading_target_prediction":
        item["extra_info"]["subtype"] = candidate.get("subtype")
        item["extra_info"]["current_block_id"] = candidate["current_block_id"]
        item["extra_info"]["candidate_label_to_block_id"] = {
            label: block_id for block_id, label in candidate["label_map"].items()
        }
        item["extra_info"]["gold_target_block_id"] = candidate["gold_answer"]
    elif candidate["task"] == "reading_order_prediction":
        item["extra_info"]["candidate_label_to_block_id"] = {
            label: block_id for block_id, label in candidate["label_map"].items()
        }
        item["extra_info"]["subtype"] = candidate.get("subtype")
    elif candidate["task"] == "parent_child_linking":
        item["extra_info"]["block_pair"] = candidate["block_pair"]
    elif candidate["task"] == "block_role_classification":
        item["extra_info"]["target_block_id"] = candidate["target_block_id"]
    return item


def build_validation_messages(candidate, item, ctx, doc_dir: Path):
    content = []
    text_part = item["prompt"][0]["content"][0]["text"]
    content.append({"type": "text", "text": text_part})
    if candidate["task"] == "reading_order_prediction":
        hidden_context = {
            "validation_only_context": True,
            "task": candidate["task"],
            "subtype": candidate.get("subtype"),
            "page_indices": candidate["page_indices"],
            "candidate_label_to_block_id": item["extra_info"].get("candidate_label_to_block_id"),
            "support_block_ids": candidate["support_block_ids"],
            "gold_answer": candidate["final_answer"],
            "support_blocks": [
                {
                    "block_id": block_id,
                    "page_index": ctx["block_by_id"][block_id]["page_index"],
                    "bbox": bbox_to_list(ctx["block_by_id"][block_id]["bbox"]),
                    "excerpt": block_text_preview(ctx["block_by_id"][block_id]),
                    "section_phase": ctx["block_to_phase"].get(block_id),
                    "section_title_id": ctx["block_to_section_title"].get(block_id),
                }
                for block_id in candidate["support_block_ids"]
            ],
        }
        content.append({"type": "text", "text": json.dumps(hidden_context, ensure_ascii=False, indent=2)})
        for page_index in candidate["page_indices"]:
            page_meta = ctx["pages_by_index"][page_index]
            image_path = page_image_path(doc_dir, page_meta)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image_data_url(image_path),
                    },
                }
            )
    else:
        for part in item["prompt"][0]["content"][1:]:
            url = part["image_url"]["url"]
            image_path = doc_dir / url
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image_data_url(image_path),
                    },
                }
            )
    system_prompt = (
        "You are verifying a scientific-document multimodal task.\n"
        "Solve the question using the provided text and images. If hidden validation-only context is present, use it only to judge whether the gold answer is reasonable and unambiguous.\n"
        "Return JSON only with keys: predicted_answer, confidence, ambiguity, short_reason."
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


def normalize_answer(task: str, answer: str):
    answer = sanitize_text(answer).strip()
    if task == "reading_order_prediction":
        answer = answer.replace(",", " > ")
        pieces = [piece.strip() for piece in re.split(r">", answer) if piece.strip()]
        return " > ".join(pieces)
    return answer


def generate_and_verify_candidate(
    candidate,
    ctx,
    doc_dir: Path,
    api_key: str,
    model: str,
    max_tokens: int,
    verify_all: bool,
    verify_reading_order: bool,
    verify_next_hop: bool,
    dry_run: bool,
    disable_question_rewrite: bool,
):
    canonical_stem = canonical_question_stem(candidate)
    answer_format = canonical_answer_format(candidate)
    if dry_run:
        generated = {
            "question_stem": canonical_stem,
            "canonical_question_stem": canonical_stem,
            "answer_format": answer_format,
            "question_rewritten": False,
        }
        item = build_final_dataset_item(candidate, generated, ctx, doc_dir)
        item["extra_info"]["verification"] = {"status": "skipped_dry_run"}
        return item

    generated = {
        "question_stem": canonical_stem,
        "canonical_question_stem": canonical_stem,
        "answer_format": answer_format,
        "question_rewritten": False,
    }
    canonical_prompt_text = build_final_prompt_text(candidate, generated, ctx)
    generated["canonical_prompt_text"] = canonical_prompt_text

    final_prompt_text = canonical_prompt_text
    if not disable_question_rewrite:
        rewritten_question_stem, rewrite_meta = rewrite_question_stem(
            canonical_question_stem_text=canonical_stem,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
        )
        generated["question_stem"] = rewritten_question_stem
        generated["rewrite_notes"] = rewrite_meta.get("rewrite_notes")
        generated["question_rewritten"] = True
        final_prompt_text = build_final_prompt_text(candidate, generated, ctx)

    item = build_final_dataset_item(candidate, generated, ctx, doc_dir, prompt_text=final_prompt_text)

    should_verify = verify_all
    if candidate["task"] == "reading_order_prediction" and verify_reading_order:
        should_verify = True
    if candidate["task"] == "next_hop_reading_target_prediction" and verify_next_hop:
        should_verify = True
    if not should_verify:
        item["extra_info"]["verification"] = {"status": "skipped_relaxed_mode"}
        return item

    validation_messages = build_validation_messages(candidate, item, ctx, doc_dir)
    verdict = call_openrouter_json(api_key, model, validation_messages, max_tokens=600, temperature=0.0)
    predicted = normalize_answer(candidate["task"], verdict.get("predicted_answer", ""))
    gold = normalize_answer(candidate["task"], candidate["final_answer"])
    item["extra_info"]["verification"] = {
        "status": "passed" if predicted == gold and not verdict.get("ambiguity") else "failed",
        "predicted_answer": predicted,
        "gold_answer": gold,
        "confidence": verdict.get("confidence"),
        "ambiguity": verdict.get("ambiguity"),
        "short_reason": verdict.get("short_reason"),
    }
    if item["extra_info"]["verification"]["status"] != "passed":
        return None
    return item


def task_candidates(document, ctx, args):
    role = extract_role_candidates(document, ctx, args.role_count)
    link = extract_linking_candidates(document, ctx, args.link_count)
    order = extract_reading_order_candidates(document, ctx, args.order_count)
    next_hop = extract_next_hop_candidates(document, ctx, args.next_hop_count)
    all_candidates = []
    for candidate in role + link + order + next_hop:
        all_candidates.append(materialize_candidate(candidate, ctx))
    return all_candidates


def output_path_for(doc_dir: Path, args):
    if args.output is not None:
        return args.output
    suffix = "dryrun" if args.dry_run else "gpt54"
    return doc_dir / "dataset" / f"layout_reading_tasks_{suffix}.jsonl"


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    document, _ = load_document(args.doc_dir)
    ctx = build_doc_context(document)
    build_text_block_context(ctx)
    configure_chat_completions_url(args.api_base_url)
    configure_openrouter_ssl(args.ca_bundle, args.insecure_ssl)
    api_key = load_api_key(args.api_key, Path(__file__))
    if not args.dry_run and not api_key:
        raise SystemExit("API key not found. Pass --api-key or set OPENAI_COMPAT_API_KEY / OPENROUTER_API_KEY.")

    candidates = task_candidates(document, ctx, args)
    # Shuffle inside the preselected pool so same task types are mixed in output.
    rng.shuffle(candidates)

    output_path = output_path_for(args.doc_dir, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emitted = []
    for candidate in candidates:
        try:
            item = generate_and_verify_candidate(
                candidate=candidate,
                ctx=ctx,
                doc_dir=args.doc_dir,
                api_key=api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                verify_all=args.verify_all,
                verify_reading_order=args.verify_reading_order,
                verify_next_hop=args.verify_next_hop,
                dry_run=args.dry_run,
                disable_question_rewrite=args.disable_question_rewrite,
            )
        except Exception as exc:
            item = None
            print(f"[skip] task={candidate['task']} reason={type(exc).__name__}: {exc}")
        if item is None:
            continue
        item["extra_info"]["question_id"] = (
            f"{args.doc_dir.name}__{candidate['task']}__{len(emitted):04d}"
        )
        emitted.append(item)

    with output_path.open("w", encoding="utf-8") as f:
        for item in emitted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(output_path)
    print(f"generated_items={len(emitted)}")


if __name__ == "__main__":
    main()
