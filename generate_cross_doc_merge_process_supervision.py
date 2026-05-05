#!/usr/bin/env python3
"""Generate process-supervision traces for D3 cross-document table-merge tasks."""

from __future__ import annotations

import argparse
import json
import gpt_client as gpt
import generate_process_supervision_common as common


TASK_NAME = "cross_doc_merge_d3"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate process-supervision traces for D3 cross-document table-merge tasks.")
    return common.add_common_args(parser).parse_args()


def build_helpful_info(item):
    extra = item.get("extra_info") or {}
    reward = item.get("reward_model") or {}
    ground_truth = reward.get("ground_truth")
    try:
        parsed_gold = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except Exception:
        parsed_gold = ground_truth
    return {
        "benchmark_name": extra.get("benchmark_name"),
        "page_a_caption": extra.get("page_a_caption"),
        "page_b_caption": extra.get("page_b_caption"),
        "ground_truth_table": parsed_gold,
    }


def task_instructions():
    return [
        "Explain how to merge two partially overlapping scientific tables into one canonical JSON table.",
        "Begin by inferring the canonical schema: which columns from the two pages refer to the same concept, and which column names should anchor the merged table.",
        "Then identify shared rows versus page-specific rows, using captions and table wording to justify row alignment when needed.",
        "Carry over exact values without inventing missing entries; when one table lacks a row or column, explain how that affects the merged result.",
        "Before the final answer, explicitly sanity-check that the merged JSON covers the needed columns, rows, and cell values.",
        "A strong trace should teach table synthesis in stages: schema alignment, row reconciliation, value transfer, and final JSON verification.",
        "If two headers or rows are semantically equivalent but phrased differently, explain the equivalence rather than silently normalizing it.",
        "The final explanation should help a student reconstruct the gold merged table, not just restate it.",
    ]


def main():
    args = parse_args()
    gpt.configure_chat_completions_url(args.api_base_url)
    ssl_mode = gpt.configure_ssl(args.ca_bundle, args.insecure_ssl)
    items = common.load_items(args.input_path)
    selected = items[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    common.init_output_files(args.output, args.log_file)

    common.log(
        f"[start] task={TASK_NAME} input={args.input_path} items={len(selected)} "
        f"dry_run={args.dry_run} workers={args.workers} model={args.model} output={args.output}"
    )
    common.log(f"[start] ssl_mode={ssl_mode}")
    common.run_generation_loop(TASK_NAME, selected, build_helpful_info, task_instructions(), args)

    common.log(f"[done] output={args.output}")


if __name__ == "__main__":
    main()
