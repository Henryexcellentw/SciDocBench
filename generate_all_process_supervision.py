#!/usr/bin/env python3
"""Batch launcher for all process-supervision task families in process_supervision_v1."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

TASK_SPECS = {
    "a1": {
        "script": "generate_layout_reading_process_supervision.py",
        "default_input": "reading_layout.jsonl",
        "default_output": "reading_layout_process.json",
    },
    "c1": {
        "script": "generate_table_logic_process_supervision.py",
        "default_input": "table_logic_1k (1).jsonl",
        "default_output": "table_logic_process.json",
    },
    "d3": {
        "script": "generate_cross_doc_merge_process_supervision.py",
        "default_input": "cross_doc_merge_1k (1).jsonl",
        "default_output": "cross_doc_merge_process.json",
    },
    "d4": {
        "script": "generate_plotqa_process_supervision.py",
        "default_input": "E1_plotqa_1k (1).jsonl",
        "default_output": "E1_plotqa_process.json",
    },
}


def log(message: str):
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run process-supervision generation for one or more task families."
    )
    parser.add_argument(
        "produced_dataset_dir",
        type=Path,
        help="Directory containing the produced dataset jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where process-supervision outputs will be written.",
    )
    parser.add_argument(
        "--messages-output-dir",
        type=Path,
        default=None,
        help="Optional directory for OpenAI-style multimodal messages outputs.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_SPECS.keys()),
        default=["a1", "c1", "d3", "d4"],
        help="Task families to run.",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used to run child scripts.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument("--api-key", default=None, help="Optional API key override.")
    parser.add_argument("--api-base-url", default=None, help="Optional API base URL override.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional max tokens override.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional start index passed to each child script.")
    parser.add_argument("--limit", type=int, default=None, help="Optional item limit passed to each child script.")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent GPT calls per child script.")
    parser.add_argument("--seed", type=int, default=42, help="Seed passed through to child scripts.")
    parser.add_argument("--dry-run", action="store_true", help="Run child scripts in dry-run mode.")
    parser.add_argument("--ca-bundle", type=Path, default=None, help="Optional CA bundle passed to child scripts.")
    parser.add_argument("--insecure-ssl", action="store_true", help="Disable SSL verification in child scripts.")
    parser.add_argument("--messages-image-root", type=Path, default=None, help="Optional root used to relativize messages image paths.")
    parser.add_argument("--messages-image-prefix", default="", help="Optional prefix prepended to messages image paths.")
    parser.add_argument("--messages-doc-loc-prefix", default=None, help="Optional doc_loc prefix passed to child scripts.")
    parser.add_argument("--messages-lang", default="en", help="Language tag for messages outputs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tasks whose output file already exists.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately if one task fails.")
    parser.add_argument("--a1-input", type=Path, default=None, help="Override input file for A1.")
    parser.add_argument("--c1-input", type=Path, default=None, help="Override input file for C1.")
    parser.add_argument("--d3-input", type=Path, default=None, help="Override input file for D3.")
    parser.add_argument("--d4-input", type=Path, default=None, help="Override input file for D4.")
    return parser.parse_args()


def input_override_for(args, task_name: str):
    return getattr(args, f"{task_name}_input")


def build_task_command(args, task_name: str):
    spec = TASK_SPECS[task_name]
    input_path = input_override_for(args, task_name) or (args.produced_dataset_dir / spec["default_input"])
    output_path = args.output_dir / spec["default_output"]
    log_path = args.output_dir / f"{output_path.stem}.log.jsonl"
    messages_output_path = None
    if args.messages_output_dir is not None:
        messages_output_path = args.messages_output_dir / f"{output_path.stem}_messages.jsonl"
    script_path = SCRIPT_DIR / spec["script"]

    cmd = [
        args.python_bin,
        str(script_path),
        str(input_path),
        "--output",
        str(output_path),
        "--log-file",
        str(log_path),
        "--start-index",
        str(args.start_index),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    if args.api_base_url:
        cmd.extend(["--api-base-url", args.api_base_url])
    if args.max_tokens is not None:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.ca_bundle is not None:
        cmd.extend(["--ca-bundle", str(args.ca_bundle)])
    if args.insecure_ssl:
        cmd.append("--insecure-ssl")
    if messages_output_path is not None:
        cmd.extend(["--messages-output", str(messages_output_path)])
    if args.messages_image_root is not None:
        cmd.extend(["--messages-image-root", str(args.messages_image_root)])
    if args.messages_image_prefix:
        cmd.extend(["--messages-image-prefix", args.messages_image_prefix])
    if args.messages_doc_loc_prefix:
        cmd.extend(["--messages-doc-loc-prefix", args.messages_doc_loc_prefix])
    if args.messages_lang:
        cmd.extend(["--messages-lang", args.messages_lang])
    return input_path, output_path, log_path, messages_output_path, cmd


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.messages_output_dir is not None:
        args.messages_output_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"[start] produced_dataset_dir={args.produced_dataset_dir} output_dir={args.output_dir} "
        f"tasks={','.join(args.tasks)} dry_run={args.dry_run}"
    )

    failures = []
    for task_name in args.tasks:
        input_path, output_path, log_path, messages_output_path, cmd = build_task_command(args, task_name)
        if args.skip_existing and output_path.exists():
            log(f"[task {task_name}] skip existing output={output_path}")
            continue
        if not input_path.exists():
            message = f"[task {task_name}] missing input={input_path}"
            log(message)
            failures.append(message)
            if args.fail_fast:
                break
            continue
        log(f"[task {task_name}] start input={input_path} output={output_path}")
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
        if result.returncode == 0:
            log(f"[task {task_name}] done output={output_path} log={log_path}")
            if messages_output_path is not None:
                log(f"[task {task_name}] messages_output={messages_output_path}")
            continue
        message = f"[task {task_name}] failed exit_code={result.returncode}"
        log(message)
        failures.append(message)
        if args.fail_fast:
            break

    if failures:
        log(f"[done] completed_with_failures={len(failures)}")
        for message in failures:
            log(f"[failure] {message}")
        raise SystemExit(1)

    log("[done] all requested tasks completed successfully.")


if __name__ == "__main__":
    main()
