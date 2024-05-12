"""
python -m inference.save_train_set_patches
"""



import json
import os
import datetime
from pathlib import Path
# from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset







data_paths = [
    "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-train-00000-of-00005.arrow",
    "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-train-00001-of-00005.arrow",
    "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-train-00002-of-00005.arrow",
    "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-train-00003-of-00005.arrow",
    "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-train-00004-of-00005.arrow",
]

train_set = []
for path in data_paths:
    dataset = load_dataset("arrow", data_files=path)["train"]
    train_set.extend(dataset)

total_instances = len(train_set)
print (f"Number of instances: {total_instances}")


output_file = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/train_set_patches.jsonl"
# keys: 'instance_id', 'model_name_or_path', 'prompt', 'full_output'
outputs = []
for i, instance in enumerate(train_set):
    # print (f"{i}/{len(train_set)}")
    instance_id = instance['instance_id']
    prompt = instance['text']
    model_name_or_path = "train_set_patches"
    full_output = instance['patch']
    outputs.append({
        'instance_id': instance_id,
        'model_name_or_path': model_name_or_path,
        'prompt': prompt,
        'full_output': full_output,
    })
# save outputs
with open(output_file, 'w') as f:
    for output in outputs:
        f.write(json.dumps(output) + '\n')
print (f"Saved to {output_file}")














































