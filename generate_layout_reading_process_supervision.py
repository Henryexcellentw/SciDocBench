#!/usr/bin/env python3
"""Generate process-supervision traces for A1 layout/reading-order tasks."""

from __future__ import annotations

import argparse
import gpt_client as gpt
import generate_process_supervision_common as common


TASK_NAME = "layout_reading_a1"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate process-supervision traces for A1 layout/reading-order tasks.")
    return common.add_common_args(parser).parse_args()


def build_helpful_info(item):
    extra = item.get("extra_info") or {}
    return {
        "task": extra.get("task"),
        "subtype": extra.get("subtype"),
        "gold_source_type": extra.get("gold_source_type"),
        "canonical_question_stem": extra.get("canonical_question_stem"),
        "current_block_id": extra.get("current_block_id"),
        "candidate_label_to_block_id": extra.get("candidate_label_to_block_id"),
        "support_block_ids": extra.get("support_block_ids"),
        "page_indices": extra.get("page_indices"),
    }


def task_instructions():
    return [
        "Explain how to solve a scientific-document layout or reading-order question.",
        "Start by identifying what kind of task this is: block role, parent-child linking, reading order, or next-hop prediction.",
        "Ground the reasoning in concrete layout cues such as page region, column structure, caption attachment, continuation behavior, and local reading flow.",
        "If candidate labels or candidate blocks exist, compare the correct answer against the strongest distractors instead of only asserting the winner.",
        "For reading-order questions, explicitly describe the ordering principle that makes the chosen order valid.",
        "Prefer concrete page-level observations such as top-left vs bottom-right placement, caption proximity, same-column continuity, heading-body relationships, and cross-page continuation cues.",
        "A strong trace should sound like a tutor teaching document parsing: first localize the relevant block(s), then interpret their roles, then justify the final relation or order.",
        "Do not invent block text or page evidence that is not visible in the prompt or helpful info.",
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
