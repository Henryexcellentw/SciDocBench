# Process Supervision V1

This directory contains the standalone scripts for generating process-supervision
data in LLaMA Factory format for the four task families already stored under
`produced_dataset/`.

Output format:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "<image><image>...\\n用户指令"},
      {"from": "gpt", "value": "过程性回答"}
    ],
    "images": [
      "/absolute/path/to/image_a.png",
      "/absolute/path/to/image_b.png"
    ]
  }
]
```

Scripts:

- `generate_layout_reading_process_supervision.py`
- `generate_table_logic_process_supervision.py`
- `generate_cross_doc_merge_process_supervision.py`
- `generate_plotqa_process_supervision.py`

Example:

```bash
python3 /Users/henrymascarpone/Downloads/1507.08818/dataset/process_supervision_v1/generate_plotqa_process_supervision.py \
  '/Users/henrymascarpone/Downloads/1507.08818/produced_dataset/E1_plotqa_1k (1).jsonl' \
  --output /Users/henrymascarpone/Downloads/1507.08818/produced_dataset/E1_plotqa_1k_process.json \
  --limit 10
```
