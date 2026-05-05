#!/usr/bin/env python3
"""End-to-end PlotQA D4 SFT generation.

This wrapper combines the two existing stages:
1. Build PlotQA-backed RL tasks by inserting PlotQA charts into SciDoc pages.
2. Convert those RL tasks into process-supervision SFT records.

The generated RL file is an intermediate artifact kept for auditability; callers
do not need to provide an existing RL dataset.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
DOWNLOAD_SCRIPT = "/mnt/shared-storage-user/mllm/shared/wushenxi/Scidocbench/IR/code/E1/download_plotqa_sample.py"
RL_BATCH_SCRIPT = "/mnt/shared-storage-user/mllm/shared/wushenxi/Scidocbench/IR/code/E1/generate_plotqa_figure_readout_tasks_batch.py"
SFT_SCRIPT = "/mnt/shared-storage-user/mllm/shared/wushenxi/Scidocbench/sft/sft_code/generate_plotqa_sft_from_scratch.py"


def log(message: str):
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/sample PlotQA, generate D4 RL tasks, then generate PlotQA SFT data."
    )
    parser.add_argument("root_dir", type=Path, help="SciDoc corpus root containing subject/document directories.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Working directory for PlotQA subset, RL, logs, and artifacts.")
    parser.add_argument("--output", type=Path, required=True, help="Final ShareGPT/LLaMA Factory SFT JSONL/JSON path.")
    parser.add_argument("--messages-output", type=Path, default=None, help="Optional OpenAI-style multimodal messages output.")
    parser.add_argument("--rl-output", type=Path, default=None, help="Optional intermediate D4 RL JSONL path.")
    parser.add_argument("--rl-log-file", type=Path, default=None, help="Optional D4 RL generation log path.")
    parser.add_argument("--sft-log-file", type=Path, default=None, help="Optional SFT generation log path.")
    parser.add_argument("--artifact-root", type=Path, default=None, help="Optional artifact root for edited pages/crops.")

    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used to run child scripts.")
    parser.add_argument("--model", default=None, help="OpenAI-compatible model for rewrite/self-check/SFT traces.")
    parser.add_argument("--api-key", default=None, help="API key. Falls back to environment or local openrouter.py.")
    parser.add_argument("--api-base-url", default=None, help="OpenAI-compatible chat completions URL.")
    parser.add_argument("--ca-bundle", type=Path, default=None, help="Optional CA bundle path.")
    parser.add_argument("--insecure-ssl", action="store_true", help="Disable HTTPS certificate verification.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call GPT in either stage.")

    parser.add_argument("--target-count", type=int, default=1000, help="Number of final RL/SFT items to generate.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap on scanned SciDoc document directories.")
    parser.add_argument("--max-items-per-doc", type=int, default=1, help="Max retained RL items per SciDoc document.")
    parser.add_argument("--plotqa-tries-per-doc", type=int, default=3, help="PlotQA candidates tried for each SciDoc document.")
    parser.add_argument("--task-types", default="single_point_readout,series_reconstruction_small", help="Comma-separated PlotQA task types.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by both stages.")

    parser.add_argument("--disable-self-check", action="store_true", help="Skip GPT self-check during RL generation.")
    parser.add_argument("--disable-question-rewrite", action="store_true", help="Skip GPT question/caption rewrite during RL generation.")
    parser.add_argument("--self-check-runs", type=int, default=5, help="Number of GPT self-check attempts per RL item.")
    parser.add_argument("--min-self-check-pass", type=int, default=1, help="Keep RL item if at least this many self-checks pass.")
    parser.add_argument("--self-check-tolerance-scale", type=float, default=1.8, help="Tolerance multiplier for self-check only.")
    parser.add_argument("--rl-max-tokens", type=int, default=1200, help="Max tokens for RL rewrite/self-check calls.")
    parser.add_argument("--sft-max-tokens", type=int, default=2200, help="Max tokens for SFT process-trace calls.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent GPT calls for the SFT stage.")

    parser.add_argument("--plotqa-local-dir", type=Path, default=None, help="Existing local PlotQA subset directory to reuse.")
    parser.add_argument("--download-plotqa", action="store_true", help="Download/sample PlotQA if --plotqa-local-dir is not provided.")
    parser.add_argument("--force-download", action="store_true", help="Refuse to reuse an existing auto-downloaded PlotQA subset.")
    parser.add_argument("--plotqa-dataset", default="achang/plot_qa", help="Hugging Face PlotQA dataset id.")
    parser.add_argument("--plotqa-split", default="train", help="Hugging Face split for download/sample.")
    parser.add_argument("--plotqa-sample-size", type=int, default=None, help="Number of local PlotQA samples to keep. Defaults to target-count.")
    parser.add_argument("--plotqa-cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--plotqa-max-source-rows", type=int, default=None, help="Optional cap on streamed PlotQA source rows.")
    parser.add_argument("--plotqa-progress-every", type=int, default=500, help="Download progress logging interval.")
    parser.add_argument("--plotqa-image-format", choices=["png", "jpg"], default="png", help="Saved PlotQA chart image format.")

    parser.add_argument("--messages-image-root", type=Path, default=None, help="Optional root used to relativize message image paths.")
    parser.add_argument("--messages-image-prefix", default="", help="Optional prefix prepended to message image paths.")
    parser.add_argument("--messages-doc-loc-prefix", default=None, help="Optional doc_loc file prefix for messages output.")
    parser.add_argument("--messages-lang", default="en", help="Language tag for messages output.")
    return parser.parse_args()


def run_command(cmd, stage_name: str):
    log(f"[{stage_name}] command={' '.join(str(part) for part in cmd)}")
    result = subprocess.run(cmd, cwd=str(DATASET_DIR))
    if result.returncode != 0:
        raise SystemExit(f"[{stage_name}] failed exit_code={result.returncode}")


def default_paths(args):
    work_dir = args.work_dir.resolve()
    rl_output = args.rl_output or (work_dir / "plotqa_d4_rl.jsonl")
    rl_log = args.rl_log_file or (work_dir / "plotqa_d4_rl.log.jsonl")
    sft_log = args.sft_log_file or (work_dir / "plotqa_d4_sft.log.jsonl")
    artifact_root = args.artifact_root or (work_dir / "plotqa_d4_artifacts")
    plotqa_local_dir = args.plotqa_local_dir or (work_dir / "plotqa_subset")
    return work_dir, rl_output, rl_log, sft_log, artifact_root, plotqa_local_dir


def build_download_command(args, plotqa_local_dir: Path):
    cmd = [
        args.python_bin,
        str(DOWNLOAD_SCRIPT),
        "--dataset",
        args.plotqa_dataset,
        "--split",
        args.plotqa_split,
        "--sample-size",
        str(args.plotqa_sample_size or args.target_count),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(plotqa_local_dir),
        "--image-format",
        args.plotqa_image_format,
        "--progress-every",
        str(args.plotqa_progress_every),
    ]
    if args.plotqa_cache_dir is not None:
        cmd.extend(["--cache-dir", str(args.plotqa_cache_dir)])
    if args.plotqa_max_source_rows is not None:
        cmd.extend(["--max-source-rows", str(args.plotqa_max_source_rows)])
    return cmd


def build_rl_command(args, plotqa_local_dir: Path, rl_output: Path, rl_log: Path, artifact_root: Path):
    cmd = [
        args.python_bin,
        str(RL_BATCH_SCRIPT),
        str(args.root_dir),
        "--output",
        str(rl_output),
        "--log-file",
        str(rl_log),
        "--artifact-root",
        str(artifact_root),
        "--target-count",
        str(args.target_count),
        "--max-items-per-doc",
        str(args.max_items_per_doc),
        "--plotqa-tries-per-doc",
        str(args.plotqa_tries_per_doc),
        "--seed",
        str(args.seed),
        "--task-types",
        args.task_types,
        "--plotqa-local-dir",
        str(plotqa_local_dir),
        "--plotqa-dataset",
        args.plotqa_dataset,
        "--plotqa-split",
        args.plotqa_split,
        "--max-tokens",
        str(args.rl_max_tokens),
        "--self-check-runs",
        str(args.self_check_runs),
        "--min-self-check-pass",
        str(args.min_self_check_pass),
        "--self-check-tolerance-scale",
        str(args.self_check_tolerance_scale),
    ]
    if args.max_docs is not None:
        cmd.extend(["--max-docs", str(args.max_docs)])
    if args.plotqa_cache_dir is not None:
        cmd.extend(["--plotqa-cache-dir", str(args.plotqa_cache_dir)])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    if args.api_base_url:
        cmd.extend(["--api-base-url", args.api_base_url])
    if args.ca_bundle is not None:
        cmd.extend(["--ca-bundle", str(args.ca_bundle)])
    if args.insecure_ssl:
        cmd.append("--insecure-ssl")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.disable_self_check:
        cmd.append("--disable-self-check")
    if args.disable_question_rewrite:
        cmd.append("--disable-question-rewrite")
    return cmd


def build_sft_command(args, rl_output: Path, sft_log: Path):
    cmd = [
        args.python_bin,
        str(SFT_SCRIPT),
        str(rl_output),
        "--output",
        str(args.output),
        "--log-file",
        str(sft_log),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(args.sft_max_tokens),
    ]
    if args.messages_output is not None:
        cmd.extend(["--messages-output", str(args.messages_output)])
    if args.messages_image_root is not None:
        cmd.extend(["--messages-image-root", str(args.messages_image_root)])
    if args.messages_image_prefix:
        cmd.extend(["--messages-image-prefix", args.messages_image_prefix])
    if args.messages_doc_loc_prefix:
        cmd.extend(["--messages-doc-loc-prefix", args.messages_doc_loc_prefix])
    if args.messages_lang:
        cmd.extend(["--messages-lang", args.messages_lang])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    if args.api_base_url:
        cmd.extend(["--api-base-url", args.api_base_url])
    if args.ca_bundle is not None:
        cmd.extend(["--ca-bundle", str(args.ca_bundle)])
    if args.insecure_ssl:
        cmd.append("--insecure-ssl")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def main():
    args = parse_args()
    if not args.root_dir.exists():
        raise SystemExit(f"Missing SciDoc root: {args.root_dir}")

    work_dir, rl_output, rl_log, sft_log, artifact_root, plotqa_local_dir = default_paths(args)
    work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.messages_output is not None:
        args.messages_output.parent.mkdir(parents=True, exist_ok=True)

    log(f"[start] root={args.root_dir}")
    log(f"[start] work_dir={work_dir}")
    log(f"[start] plotqa_local_dir={plotqa_local_dir}")
    log(f"[start] rl_output={rl_output}")
    log(f"[start] sft_output={args.output}")
    if args.messages_output is not None:
        log(f"[start] messages_output={args.messages_output}")

    if args.plotqa_local_dir is not None:
        if not plotqa_local_dir.exists():
            raise SystemExit(f"--plotqa-local-dir does not exist: {plotqa_local_dir}")
        log("[download] skipped because --plotqa-local-dir was provided")
    elif plotqa_local_dir.exists() and not args.force_download:
        log("[download] skipped because auto PlotQA subset already exists")
    else:
        if not args.download_plotqa and not plotqa_local_dir.exists():
            raise SystemExit(
                "No local PlotQA subset found. Pass --download-plotqa to create one, "
                "or pass --plotqa-local-dir to reuse an existing subset."
            )
        run_command(build_download_command(args, plotqa_local_dir), "download")

    run_command(build_rl_command(args, plotqa_local_dir, rl_output, rl_log, artifact_root), "rl")
    run_command(build_sft_command(args, rl_output, sft_log), "sft")

    log(f"[done] rl_output={rl_output}")
    log(f"[done] sft_output={args.output}")
    if args.messages_output is not None:
        log(f"[done] messages_output={args.messages_output}")


if __name__ == "__main__":
    main()
