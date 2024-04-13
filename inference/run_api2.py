#!/usr/bin/env python3

"""
python -m inference.run_api2
python inference/run_api2.py because of how things get imported...fixed it.. 
python -m inference.run_api2 --dataset_name_or_path /home/chris_cohere_ai/SWE-bench-stuff/tasks/test_set/swe-bench.json --model_name_or_path command-r --output_dir /home/chris_cohere_ai/SWE-bench-stuff/outputs

This python script is designed to run inference on a dataset using either the OpenAI or Anthropic API, depending on the model specified. 
It sorts instances by length and continually writes the outputs to a specified file, so that the script can be stopped and restarted without losing progress.
"""

import json
import os
import time
import dotenv
import traceback
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import openai
import tiktoken
from anthropic import HUMAN_PROMPT, AI_PROMPT, Anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from datasets import load_dataset, load_from_disk

from argparse import ArgumentParser
import logging


from inference.run_api import MODEL_LIMITS, parse_model_args, openai_inference, anthropic_inference
from inference.make_datasets.utils import extract_diff

from swebench.harness.colours import blue

import concurrent.futures

import cohere


# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# logger = logging.getLogger(__name__)
# dotenv.load_dotenv()


logging.getLogger().setLevel(logging.WARNING)






def cohere_inference(
    test_dataset,
    model_name_or_path,
    output_file,
    model_args,
    existing_ids,
    max_cost,
):
    

    

    
    # # im going to save a json with instance_id, model_name_or_path and model_patch
    # # then i will use that json to run the evaluation script
    # import pandas as pd
    # output_path = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/provided_patch.json"
    # json_data = []
    # for row in test_dataset:
    #     json_data.append({'instance_id': row['instance_id'], 
    #                       'model_name_or_path': "their_provided_patch", 
    #                       'model_patch': row['patch']})
    # pd.DataFrame(json_data).to_json(output_path, orient='records')
    # print (f"Saved to {output_path}")
    # fsda






    # encoding = tiktoken.encoding_for_model('gpt-35-turbo')
    # print (encoding)
    # fasd




    # # prin text column
    # print (test_dataset.column_names)
    # # fasd
    # # [r]
    # print ('\nTEXT')
    # print (test_dataset["text"][0])
    # print ()
    # print ()
    # print ("\nPATCH")
    # print (test_dataset["patch"][0])
    # print ()
    # print ()
    # # print ("\nTEST PATCH")
    # # print (test_dataset["test_patch"][0])
    # print ("\n FAIL TO PASS")
    # print (test_dataset["FAIL_TO_PASS"][0])
    # print ("\n PASS TO PASS")
    # print (test_dataset["PASS_TO_PASS"][0])
    # fadsfa



    
    print (len(test_dataset))


    # remove instances that are too long
    cohere_tokenize = lambda x: len(x) / 3.4
    test_dataset = test_dataset.filter(
        lambda x: cohere_tokenize(x["text"]) <= 120000,
        desc="Filtering",
        load_from_cache_file=False,
    )
    print(f"Filtered to {blue(len(test_dataset))} instances due to length\n")



    def get_responses(client, model_name, message):
        response = client.chat(
            message=message,
            temperature=0,
            model=model_name,
        )
        return response.__dict__

    api_key = os.environ.get("COHERE_API_KEY", None)
    cohere_client = cohere.Client(api_key)
    # set the client to the cohere client
    get_responses_cohere = lambda x: get_responses(cohere_client, model_name_or_path, x)

    n_datapoints = len(test_dataset)
    batch_size = 8
    batch = []
    for i in tqdm(range(n_datapoints)):

        batch.append(test_dataset[i])
        if len(batch) == batch_size or i == n_datapoints-1:
            text_batch = [datum['text'] for datum in batch]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                responses = list(executor.map(get_responses_cohere, text_batch))

            for datum, response in zip(batch, responses):
                completion = response['text']
                # print (response.keys())
                output_dict = {
                    "instance_id": datum["instance_id"], 
                    "model_name_or_path": model_name_or_path,
                    # "text": datum['text'],
                    "full_output": completion,
                    "model_patch": extract_diff(completion),
                }

                with open(output_file, "a+") as f:
                    f.write(json.dumps(output_dict) + "\n")
            batch = []

            # break




def main(
    dataset_name_or_path,
    split,
    model_name_or_path,
    shard_id,
    num_shards,
    output_dir,
    model_args,
    max_cost,
):
    # if shard_id is None and num_shards is not None:
    #     logger.warning(
    #         f"Received num_shards={num_shards} but shard_id is None, ignoring"
    #     )
    # if shard_id is not None and num_shards is None:
    #     logger.warning(f"Received shard_id={shard_id} but num_shards is None, ignoring")
    # model_args = parse_model_args(model_args)
    
    # if "checkpoint" in Path(model_name_or_path).name:
    #     model_nickname = Path(model_name_or_path).parent.name
    # else:
    #     model_nickname = Path(model_name_or_path).name

    # print ('\n HEHEHEHEHEHEHEH \n')

    model_nickname = model_name_or_path
    output_file = f"{model_nickname}__{dataset_name_or_path.split('/')[-1]}__{split}"
    # if shard_id is not None and num_shards is not None:
    #     output_file += f"__shard-{shard_id}__num_shards-{num_shards}"
    output_file = Path(output_dir, output_file + ".jsonl")
    print(f"\nWill write to\n {blue(output_file)}\n")
    existing_ids = set()
    if os.path.exists(output_file):
        # TODO: dont redo the ones already done.
        exit()
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                existing_ids.add(instance_id)
    print(f"Read {len(existing_ids)} already completed ids from {output_file}")

    # print ('\n 111111111111111111111 \n')

    if Path(dataset_name_or_path).exists():
        dataset = load_from_disk(dataset_name_or_path)
    else:
        print ('\n loading dataset')
        dataset = load_dataset(dataset_name_or_path)

    print ('dataset loaded \n')

    if not split in dataset:
        raise ValueError(f"Invalid split {split} for dataset {dataset_name_or_path}")
    dataset = dataset[split]
    print (dataset.column_names)
    print ()
    lens = np.array(list(map(len, dataset["text"])))

    print ('\n 2222222222222222222222222 \n')
    print (len(lens))


    dataset = dataset.select(np.argsort(lens))
    if len(existing_ids) > 0:
        dataset = dataset.filter(
            lambda x: x["instance_id"] not in existing_ids,
            desc="Filtering out existing ids",
            load_from_cache_file=False,
        )
    # if shard_id is not None and num_shards is not None:
    #     dataset = dataset.shard(num_shards, shard_id, contiguous=True)
    inference_args = {
        "test_dataset": dataset,
        "model_name_or_path": model_name_or_path,
        "output_file": output_file,
        "model_args": model_args,
        "existing_ids": existing_ids,
        "max_cost": max_cost,
    }

    cohere_inference(**inference_args)

    # if model_name_or_path.startswith("claude"):
    #     anthropic_inference(**inference_args)
    # elif model_name_or_path.startswith("gpt"):
    #     openai_inference(**inference_args)
    # else:
    #     raise ValueError(f"Invalid model name or path {model_name_or_path}")
    print(f"Done!")

















if __name__ == "__main__":

    # dataset_name_or_path = "/home/chris_cohere_ai/SWE-bench-stuff/tasks/test_set/swe-bench.json"
    # dataset_name_or_path = "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench/"
    # dataset_name_or_path = "princeton-nlp/SWE-bench"
    dataset_name_or_path = "princeton-nlp/SWE-bench_oracle"
    
    
    output_dir = "/home/chris_cohere_ai/SWE-bench-stuff/outputs"
    model_name = "command-r"
    
    main(dataset_name_or_path,
            "test",
            model_name,
            None,
            None,
            output_dir,
            None,
            None)


    # parser = ArgumentParser(description=__doc__)
    # parser.add_argument(
    #     "--dataset_name_or_path",
    #     type=str,
    #     required=True,
    #     help="HuggingFace dataset name or local path",
    # )
    # parser.add_argument(
    #     "--split",
    #     type=str,
    #     default="test",
    #     help="Dataset split to use",
    # )
    # parser.add_argument(
    #     "--model_name_or_path",
    #     type=str,
    #     help="Name of API model. Update MODEL* constants in this file to add new models.",
    #     choices=sorted(list(MODEL_LIMITS.keys())),
    # )
    # parser.add_argument(
    #     "--shard_id",
    #     type=int,
    #     default=None,
    #     help="Shard id to process. If None, process all shards.",
    # )
    # parser.add_argument(
    #     "--num_shards",
    #     type=int,
    #     default=None,
    #     help="Number of shards. If None, process all shards.",
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Path to the output file.",
    # )
    # parser.add_argument(
    #     "--model_args",
    #     type=str,
    #     default=None,
    #     help="List of model arguments separated by commas. (e.g. 'top_p=0.95,temperature=0.70')",
    # )
    # parser.add_argument(
    #     "--max_cost",
    #     type=float,
    #     default=None,
    #     help="Maximum cost to spend on inference.",
    # )
    # args = parser.parse_args()
    # main(**vars(args))
