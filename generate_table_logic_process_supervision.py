#!/usr/bin/env python3
"""Generate process-supervision traces for C1 table logic consistency tasks."""

from __future__ import annotations

import argparse
import gpt_client as gpt
import generate_process_supervision_common as common


TASK_NAME = "table_logic_c1"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate process-supervision traces for C1 table logic consistency tasks.")
    return common.add_common_args(parser).parse_args()


def build_helpful_info(item):
    extra = item.get("extra_info") or {}
    return {
        "claim_type": extra.get("claim_type"),
        "fact_summary": extra.get("fact_summary"),
        "caption_text": extra.get("caption_text"),
        "final_corrupted_statement": extra.get("final_corrupted_statement"),
        "candidate_label_to_option": extra.get("candidate_label_to_option"),
        "gold_label": extra.get("gold_label"),
        "supporting_cell_ids": extra.get("supporting_cell_ids"),
    }


def task_instructions():
    return [
        "Explain how to verify a table-related claim against the evidence in the page, table, and caption.",
        "First restate what part of the claim is suspicious: entity, metric, value, trend, ranking, or aggregate.",
        "Then align the statement with the supporting table evidence and the fact summary to identify the exact inconsistency.",
        "Use the candidate options to show why the correct repair matches the evidence better than the nearest distractors.",
        "If supporting cells are provided, treat them as teacher-side anchors for the reasoning but still explain the logic in natural language.",
        "A strong trace should identify the compared entities and metrics before reading the numbers, then use the numbers to validate or reject each option.",
        "When the error is about ranking, trend, percentage, or aggregate, explain the required arithmetic or comparison explicitly instead of only naming the winner.",
        "Keep the reasoning faithful to the visible table evidence and the provided factual support.",
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
