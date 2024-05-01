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

# import sys
# sys.path.insert(0, '.')
from display.utils import get_model_report3, get_tests_results, get_log_path


# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# logger = logging.getLogger(__name__)
# dotenv.load_dotenv()


logging.getLogger().setLevel(logging.WARNING)














def cohere_inference(
    test_dataset,
    model_name_or_path,
    output_file,
    past_output,
    past_logs
):
    
    print (f'Original dataset length: {len(test_dataset)}\n')

    # Resolved instances that gold patch cant solve
    print ('Resolved instances that gold patch cant solve')
    gold_resolved_path = "/home/chris_cohere_ai/SWE-bench-stuff/tasks/resolved_instances/swe-bench-oracle-resolved.json"
    with open(gold_resolved_path, "r") as f:
        resolved_instances = json.load(f)
    print (len(resolved_instances))

    datapoints = {}
    for i in range(len(test_dataset)):
        # print (test_dataset[i]['instance_id'])
        if test_dataset[i]['instance_id'] in resolved_instances:
            # print ('resolved')
            datapoints[test_dataset[i]['instance_id']] = test_dataset[i]
    print (len(datapoints))
    # print (datapoints[0].keys())
    instance_ids_todo = list(datapoints.keys())




    # Remove instances that have already been solved
    print ('Remove instances that have already been solved')
    datapoints2 = {}
    for log_file in os.listdir(past_logs):
        instance_id = log_file.split(".")[0]
        if instance_id not in instance_ids_todo:
            continue

        tests_PASS_TO_PASS = datapoints[instance_id]["PASS_TO_PASS"]
        tests_FAIL_TO_PASS = datapoints[instance_id]["FAIL_TO_PASS"]
        log_path = os.path.join(past_logs, log_file)

        log_content, result = get_model_report3(log_path=log_path)
        report = get_tests_results(log_content, instance_id, tests_PASS_TO_PASS, tests_FAIL_TO_PASS)
        if len(report["PASS_TO_PASS"]["failure"]) > 0 or len(report["FAIL_TO_PASS"]["failure"]) > 0:
            datapoints2[instance_id] = datapoints[instance_id]
        else:
            print ('heheheheh')

        datapoints2[instance_id]['log_content'] = log_content
            
    print (len(datapoints2))

    # Collect previous outputs
    print ('Collect previous outputs')
    with open(past_output, "r") as f:
        for line in f:
            data = json.loads(line)
            # print (data.keys())
            # fdasdf
            instance_id = data["instance_id"]
            if instance_id in datapoints2:
                datapoints2[instance_id]["model_output"] = data["model_patch"]


    # Create prompts here using past prompts and errors. 
    print ('Create prompts here using past prompts and errors.')
    for instance_id, datapoint in datapoints2.items():
        prompt = datapoint["text"]
        prompt += "\n\nThis is what the model outputted last time:\n"
        prompt += datapoint["model_output"]
        prompt += "\n\nThis is the error message:\n"
        prompt += datapoint["log_content"][-2000:]
        prompt += "\n\nPlease fix the error and provide a patch that passes the tests."
        datapoint["prompt"] = prompt


    # Remove instances that are too long
    print ('Remove instances that are too long')
    datapoints = []
    for instance_id, datapoint in datapoints2.items():
        if len(datapoint["prompt"]) <= 120000 * 3.4:
            # datapoints[instance_id] = datapoint
            datapoints.append(datapoint)
    print (len(datapoints))

    # cohere_tokenize = lambda x: len(x) / 3.4
    # test_dataset = test_dataset.filter(
    #     lambda x: cohere_tokenize(x["text"]) <= 120000,
    #     desc="Filtering",
    #     load_from_cache_file=False,
    # )
    # print(f"Filtered to {blue(len(test_dataset))} instances due to length\n")

    # prompts = [datum['text'] for datum in test_dataset]
    # fasdafsd


    def get_responses(client, model_name, datapoint):
        prompt = datapoint["prompt"]
        response = client.chat(
            message=prompt,
            temperature=0,
            model=model_name,
        )
        return response.__dict__

    api_key = os.environ.get("COHERE_API_KEY", None)
    cohere_client = cohere.Client(api_key)
    # set the client to the cohere client
    get_responses_cohere = lambda x: get_responses(cohere_client, model_name_or_path, x)

    n_datapoints = len(datapoints)
    batch_size = 8
    # batch = []
    batch = []
    for i in tqdm(range(n_datapoints)):

        # batch.append(test_dataset[i])
        batch.append(datapoints[i])
        if len(batch) == batch_size or i == n_datapoints-1:
            # text_batch = [datum['text'] for datum in batch]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                responses = list(executor.map(get_responses_cohere, batch))

            for datum, response in zip(batch, responses):
                completion = response['text']
                # print (response.keys())
                output_dict = {
                    "instance_id": datum["instance_id"], 
                    "model_name_or_path": model_name_or_path,
                    "prompt": datum["text"],
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
    model_name_suffix,
    output_dir,
    past_output,
    past_logs,
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

    model_nickname = model_name_or_path + model_name_suffix
    output_file = f"{model_nickname}__{dataset_name_or_path.split('/')[-1]}__{split}"
    # if shard_id is not None and num_shards is not None:
    #     output_file += f"__shard-{shard_id}__num_shards-{num_shards}"
    output_file = Path(output_dir, output_file + ".jsonl")
    print(f"\nWill write to\n {blue(output_file)}\n")
    existing_ids = set()
    if os.path.exists(output_file):
        # # TODO: dont redo the ones already done.
        # exit()
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                existing_ids.add(instance_id)
    print(f"Read {blue(len(existing_ids))} already completed ids from {output_file}")



    # Load dataset
    if Path(dataset_name_or_path).exists():
        dataset = load_from_disk(dataset_name_or_path)
    else:
        print ('\n loading dataset')
        dataset = load_dataset(dataset_name_or_path)
    print ('dataset loaded \n')



    dataset = dataset[split]
    # Sort by length
    lens = np.array(list(map(len, dataset["text"])))
    dataset = dataset.select(np.argsort(lens))

    print (dataset.column_names)
    print ()
    


    # Filter out existing ids
    if len(existing_ids) > 0:
        dataset = dataset.filter(
            lambda x: x["instance_id"] not in existing_ids,
            desc="Filtering out existing ids",
            load_from_cache_file=False,
        )
    print(f"Filtered to {blue(len(dataset))} instances\n")


    cohere_inference(
            test_dataset=dataset,
            model_name_or_path=model_name_or_path,
            output_file=output_file,
            past_output=past_output,
            past_logs=past_logs
    )

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
    # model_name = "command-r"
    # model_name = "command-r-plus"
    model_name = "command-r-plus" #-round2"
    model_name_suffix = "_round2"

    past_output = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/command-r-plus__SWE-bench_oracle__test.jsonl"
    past_logs = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir/command-r-plus"
    
    main(dataset_name_or_path,
            "test",
            model_name,
            model_name_suffix,
            output_dir,
            past_output,
            past_logs)

























    
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
