#!/usr/bin/env python3
"""Generate process-supervision traces for D4 PlotQA-backed figure readout tasks."""

from __future__ import annotations

import argparse
import gpt_client as gpt
import generate_process_supervision_common as common


TASK_NAME = "plotqa_figure_readout_d4"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate process-supervision traces for D4 PlotQA-backed figure readout tasks.")
    return common.add_common_args(parser).parse_args()


def build_helpful_info(item):
    extra = item.get("extra_info") or {}
    reward = item.get("reward_model") or {}
    return {
        "task_type": extra.get("task_type"),
        "chart_type": extra.get("chart_type"),
        "chart_title": extra.get("chart_title"),
        "x_axis_label": extra.get("x_axis_label"),
        "y_axis_label": extra.get("y_axis_label"),
        "series_name": extra.get("series_name"),
        "target_x_raw": extra.get("target_x_raw"),
        "target_x_labels": extra.get("target_x_labels"),
        "target_x_mode": extra.get("target_x_mode"),
        "point_count": extra.get("point_count"),
        "tolerance_abs": extra.get("tolerance_abs"),
        "gold_answer_payload": reward.get("ground_truth"),
    }


def task_instructions():
    return [
        "Explain how to read exact or near-exact values from a scientific chart embedded in a paper page.",
        "Start by locating the target series from the legend, marker style, or caption, then explain how to identify the requested x-position or the full left-to-right sequence.",
        "Describe how the y-values are estimated from the axis, grid, or local geometry rather than jumping straight to the answer.",
        "If the task asks for a full sequence, enumerate the reading order explicitly and keep the values aligned with that order.",
        "Use the provided tolerance only as grading context; still aim to explain the cleanest and most defensible value-reading procedure.",
        "Include a short final check that the numeric answer is in the requested JSON format.",
        "A strong trace should sound like careful chart reading: identify the correct series, anchor the target x-location, read the vertical position against the axis, then refine to the best numerical estimate.",
        "If multiple series are present, explicitly mention how you avoid confusing the target series with the nearby alternatives.",
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
