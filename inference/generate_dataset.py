"""
python -m inference.generate_dataset
"""


import json
import os
import datetime
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset

from argparse import ArgumentParser
import logging

from swebench.harness.colours import blue
import concurrent.futures


from inference.make_datasets.utils import extract_diff

from display.utils import get_model_report3, get_tests_results, get_log_path

from inference.cohere_samplers import cohere_inference, Fax_Sampler
import cohere

logging.getLogger().setLevel(logging.WARNING)






# def filter_examples(dataset, output_file, max_length, past_output, past_logs):

#     # Convert dataset to dict where key is instance_id
#     dataset2 = {}
#     for i in range(len(dataset)):
#         dataset2[dataset[i]['instance_id']] = dataset[i]
#     print (f'Original dataset length: {len(dataset2)}')

#     # Only keep instances that gold patch solves AND are less than 40k tokens, 818 of them
#     test_instances_path = "/home/chris_cohere_ai/SWE-bench-stuff/instance_ids/test_lite_cc_750.json"
#     test_instances = json.load(open(test_instances_path))
#     dataset2 = {k: v for k, v in dataset2.items() if k in test_instances}
#     print (f"Filtered to {blue(len(dataset2))}: solved by gold and less than max_length tokens")



#     # Filter out existing ids
#     # if len(existing_ids) > 0:
#     #     dataset = dataset.filter(
#     #         lambda x: x["instance_id"] not in existing_ids,
#     #         desc="Filtering out existing ids",
#     #         load_from_cache_file=False,
#     #     )
    
    
#     # if not overwrite and output_file.exists():
#     if output_file.exists():
#         existing_ids = set()
#         with open(output_file) as f:
#             for line in f:
#                 data = json.loads(line)
#                 instance_id = data["instance_id"]
#                 existing_ids.add(instance_id)
#         print(f"Read {blue(len(existing_ids))} already completed ids from {output_file}")

#         dataset2 = {k: v for k, v in dataset2.items() if k not in existing_ids}
#         print(f"Filtered to {blue(len(dataset2))}, already generated instances")



#     # Remove instances that gold patch cant solve
#     gold_resolved_path = "/home/chris_cohere_ai/SWE-bench-stuff/tasks/resolved_instances/swe-bench-oracle-resolved.json"
#     with open(gold_resolved_path, "r") as f:
#         resolved_instances = json.load(f)
#     print (f'Instances that gold patch can solve: {len(resolved_instances)}')
#     datapoints2 = {k: v for k, v in dataset2.items() if k in resolved_instances}
#     print (f"Filtered to {blue(len(datapoints2))} instances that gold patch can solve")


#     # datapoints = {}
#     # for i in range(len(dataset2)):
#     #     # print (test_dataset[i]['instance_id'])
#     #     if dataset2[i]['instance_id'] in resolved_instances:
#     #         # print ('resolved')
#     #         datapoints[dataset2[i]['instance_id']] = dataset2[i]
#     # print (len(datapoints))
#     # # print (datapoints[0].keys())
#     # instance_ids_todo = list(datapoints.keys())



#     if past_output:
#         # Remove instances that have already been solved
#         print ('Remove instances that have already been solved')
#         datapoints2 = {}
#         for log_file in os.listdir(past_logs):
#             instance_id = log_file.split(".")[0]
#             if instance_id not in instance_ids_todo:
#                 continue

#             tests_PASS_TO_PASS = datapoints[instance_id]["PASS_TO_PASS"]
#             tests_FAIL_TO_PASS = datapoints[instance_id]["FAIL_TO_PASS"]
#             log_path = os.path.join(past_logs, log_file)

#             log_content, result = get_model_report3(log_path=log_path)
#             report = get_tests_results(log_content, instance_id, tests_PASS_TO_PASS, tests_FAIL_TO_PASS)
#             if len(report["PASS_TO_PASS"]["failure"]) > 0 or len(report["FAIL_TO_PASS"]["failure"]) > 0:
#                 datapoints2[instance_id] = datapoints[instance_id]
#             else:
#                 print ('heheheheh')

#             datapoints2[instance_id]['log_content'] = log_content
                
#         print (len(datapoints2))



#         # Collect previous outputs
#         print ('Collect previous outputs')
#         with open(past_output, "r") as f:
#             for line in f:
#                 data = json.loads(line)
#                 # print (data.keys())
#                 # fdasdf
#                 instance_id = data["instance_id"]
#                 if instance_id in datapoints2:
#                     datapoints2[instance_id]["model_output"] = data["model_patch"]
#                     datapoints2[instance_id]["prompt"] = data["prompt"]


#         # Create prompts here using past prompts and errors. 
#         print ('Create prompts here using past prompts and errors.')
#         for instance_id, datapoint in datapoints2.items():
#             if "prompt" in datapoint:
#                 prompt = datapoint["prompt"]
#             else:
#                 prompt = datapoint["text"]
#             prompt += "\n\nThis is what you outputted last time:\n"
#             prompt += f'<output>\n{datapoint["model_output"]}\n</output>'
#             prompt += "\n\nThis is the error message:\n"
#             prompt += f'<logs>\n{datapoint["log_content"][-2000:]}\n</logs>'
#             prompt += "\n\nPlease fix the error and provide a patch that passes the tests.\n"
#             datapoint["prompt"] = prompt
#             # print (prompt)
#             # fdsaf

#     else:
#         # datapoints2 = {}
#         # for instance_id, datapoint in dataset2.items():
#         #     datapoint["prompt"] = datapoint["text"]
#         #     datapoints2[instance_id] = datapoint
#         for instance_id, datapoint in datapoints2.items():
#             datapoint["prompt"] = datapoint["text"]

#     # co = cohere.Client(os.environ['COHERE_API_KEY'])
#     # response = co.tokenize(text="tokenize me! :D", model="command")  # optional
#     # print(len(response.tokens))


#     # Remove instances that are too long
#     # For agent model, it can become too long..need better fix. .
#     # print ('Remove instances that are too long')
#     datapoints = []
#     for instance_id, datapoint in datapoints2.items():
#         # prompt_len = len(co.tokenize(text=datapoint["prompt"], model="command").tokens)
#         # if prompt_len <= max_length-3:
#         if len(datapoint["prompt"]) <= max_length * 3:
#             datapoints.append(datapoint)
#     # print (len(datapoints))
#     print (f"Filtered to {blue(len(datapoints))} instances due to length\n")
#     # fadsfsd




#     # cohere_tokenize = lambda x: len(x) / 3.4
#     # test_dataset = test_dataset.filter(
#     #     lambda x: cohere_tokenize(x["text"]) <= 120000,
#     #     desc="Filtering",
#     #     load_from_cache_file=False,
#     # )
#     # print(f"Filtered to {blue(len(test_dataset))} instances due to length\n")

#     # prompts = [datum['text'] for datum in test_dataset]
#     # fasdafsd
#     return datapoints



# def get_lengths(train_set):

#     cohere_key = os.environ['COHERE_API_KEY']
#     co = cohere.Client(cohere_key)

#     instance_lengths_path = "/home/chris_cohere_ai/SWE-bench-stuff/instance_ids/train_set_lengths.json"
#     lengths = {}
#     for instance in train_set:
#         # print (instance.keys())
#         instance_id = instance['instance_id']
#         text = instance['text']

#         response = co.tokenize(text=text, model="command")
#         # print(len(response.tokens))
#         lengths[instance_id] = len(response.tokens)

#     # Save lengths
#     with open(instance_lengths_path, "w") as f:
#         json.dump(lengths, f)
#     print (f"Saved to {instance_lengths_path}")
    





def get_responses(client, model_name, prompt):
    try:
        response = client.chat(
            message=prompt,
            temperature=1.0,
            model=model_name,
        )
    except Exception as e:
        print (f"Error: {e}")
        print (response)
        fadsfasd
    return response.__dict__



def generate_completions(prompt, model, n_completions):

    batch = [prompt] * n_completions

    if model in ["command-r", "command-r-plus"]:
        # Prod
        api_key = os.environ.get("COHERE_API_KEY", None)
        cohere_client = cohere.Client(api_key)

        # Staging
        # api_key = os.environ.get("COHERE_STG_API_KEY")
        # cohere_client = cohere.Client(base_url='https://stg.api.cohere.ai', api_key=api_key)

        get_responses_cohere = lambda x: get_responses(cohere_client, model, x)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            responses = list(executor.map(get_responses_cohere, batch))
        completions = [response['text'] for response in responses]
        finish_reasons = [response['finish_reason'] for response in responses]

    else:
        sampler = Fax_Sampler(model_path)
        completions = sampler.sample(batch, temp=1)




    return completions, finish_reasons





















    















if __name__ == "__main__":
    """
    python -m inference.generate_dataset
    """

    date = datetime.datetime.now().strftime("%Y_%m_%d")
    
    
    # run_name = f"command-r-plus-{date}-test-set"
    # model = "command-r-plus"
    # prev_outputs_dir = "/home/chris_cohere_ai/SWE-bench-stuff/generated_completions/command-r-plus-2024_05_12-test-set"

    run_name = f"35B_finetune-{date}-test-set"
    model = "100.96.123.96:8000"
    prev_outputs_dir = ""



    
    # Output dir
    if prev_outputs_dir:
        run_dir = prev_outputs_dir
    else:
        generations_dir = "/home/chris_cohere_ai/SWE-bench-stuff/generated_completions"
        run_dir = os.path.join(generations_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

    print (f"Run dir: {blue(run_dir)}")


    data_paths = [
        # "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-train-00000-of-00005.arrow",
        # "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-train-00001-of-00005.arrow",
        # "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-train-00002-of-00005.arrow",
        # "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-train-00003-of-00005.arrow",
        # "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-train-00004-of-00005.arrow",
    
        # test set
        "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-test.arrow"

    ]

    train_set = []
    for path in data_paths:
        dataset = load_dataset("arrow", data_files=path)["train"]
        train_set.extend(dataset)

    total_instances = len(train_set)
    print (f"Number of instances: {blue(total_instances)}")


    # # Load lengths
    # lengths_path = "/home/chris_cohere_ai/SWE-bench-stuff/instance_ids/train_set_lengths.json"
    # with open(lengths_path, "r") as f:
    #     lengths = json.load(f)
    # print (f"Loaded lengths from {lengths_path}")
    # count = 0 
    # print (len(lengths))
    # # print (lengths[0])
    # new_set = []
    # for instance, length in lengths.items():
    #     if length < 35000:
    #         new_set.append(instance)
    # print (f"{len(new_set)}/{len(lengths)}")
    # train_set = [instance for instance in train_set if instance['instance_id'] in new_set]


    # Only keep instances that gold patch solves AND are less than 40k tokens, 750 of them
    test_instances_path = "/home/chris_cohere_ai/SWE-bench-stuff/instance_ids/test_lite_cc_750.json"
    test_instances = json.load(open(test_instances_path))
    train_set = [instance for instance in train_set if instance['instance_id'] in test_instances]
    print (f"Filtered to {blue(len(train_set))}: solved by gold and less than max_length tokens")
    # fdasfsad

    # Remove instances that have already been solved
    if prev_outputs_dir:
        completed_ids = []
        for file in os.listdir(prev_outputs_dir):
            instance_id = file.split(".")[0]
            completed_ids.append(instance_id)

        train_set = [instance for instance in train_set if instance['instance_id'] not in completed_ids]
        print (f"Filtered to {blue(len(train_set))}, due to already generated instances")




    #####################################################################
    print ()
    print ()
    print (train_set[0].keys())
    # Get counts for each repo
    repo_counts = {}
    for instance in train_set:
        repo = instance['repo']
        if repo in repo_counts:
            repo_counts[repo] += 1
        else:
            repo_counts[repo] = 1
    # sort
    repo_counts = {k: v for k, v in sorted(repo_counts.items(), key=lambda item: item[1], reverse=True)}
    for repo, count in repo_counts.items():
        print (f"{repo}: {count}")
    print ()

    # for instance in train_set:
    #     if 'pandas' in instance['repo']:
    #         print ()
    #         print (instance['instance_id'])
    #         print (instance['base_commit'])
    #         print () 
    #         fadsf
    # fsadf
    #####################################################################






    n_completions = 10
    for i, instance in enumerate(train_set):
        
        instance_id = instance['instance_id']
        text = instance['text']
        print (f"{i}/{len(train_set)} - {instance_id}")

        completions, finish_reasons = generate_completions(text, model, n_completions)
        # for c in completions:
        #     print (c)
        #     print ('====\nextract diff')
        #     print (extract_diff(c))
        #     print ('----------------------------\n\n')
        # fsdaafds

        instance_path = os.path.join(run_dir, f"{instance_id}.json")
        instance['completions'] = completions
        instance['finish_reasons'] = finish_reasons
        with open(instance_path, "w") as f:
            json.dump(instance, f)







































    fadsfa


    # dataset_name_or_path = "/home/chris_cohere_ai/SWE-bench-stuff/tasks/test_set/swe-bench.json"
    # dataset_name_or_path = "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench/"
    # dataset_name_or_path = "princeton-nlp/SWE-bench"

    # model_name = "command-r"
    # model_name = "command-r-plus"
     #-round2"
    # model_name_suffix = "_round2"
    # model_name_suffix = "_agent2"


    # past_output = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/command-r-plus__SWE-bench_oracle__test.jsonl"
    # past_logs = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir/command-r-plus"
    

    # model_name = "command-r-plus"
    # model_name_suffix = "_agent3"
    # past_output = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/command-r-plus_agent2__SWE-bench_oracle__test.jsonl"
    # past_logs = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir/command-r-plus_agent2"


    # model_name = "finetuned_35B"
    # model_path = "100.96.123.96:8000"
    # model_name_suffix = ""
    # past_output = ""
    # past_logs = ""
    # max_length = 40000
    # overwrite = True


    # model_name = "test"
    # model_path = "100.96.123.96:8000"
    # model_name_suffix = ""
    # past_output = ""
    # past_logs = ""
    # max_length = 40000
    # overwrite = True




    # run_name = f"command-r-plus_{date}"
    # model_path = "command-r-plus"
    # past_output = ""
    # past_logs = ""
    # max_length = 40000
    # overwrite = False


    # run_name = f"command-r_{date}"
    # model_path = "command-r"
    # past_output = ""
    # past_logs = ""
    # max_length = 34000
    # overwrite = False



    # run_name = f"test"
    # model_path = "command-r"
    # past_output = ""
    # past_logs = ""
    # max_length = 33300
    # overwrite = True


    run_name = "finetuned_35B"
    model_path = "100.96.123.96:8000"
    past_output = ""
    past_logs = ""
    max_length = 33300
    overwrite = False



    # dataset_name_or_path = "princeton-nlp/SWE-bench_oracle"
    output_dir = "/home/chris_cohere_ai/SWE-bench-stuff/outputs"

    # main(   run_name,
    #         model_path,
    #         max_length,
    #         output_dir,
    #         past_output,
    #         past_logs,
    #         overwrite)







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

    split = 'test'
    dataset_name = 'SWE-bench_oracle'

    # model_nickname = model_name + model_name_suffix
    output_file = f"{run_name}__{dataset_name.split('/')[-1]}__{split}"
    output_file = Path(output_dir, output_file + ".jsonl")
    print(f"\nWill write to\n {blue(output_file)}\n")

    # show_outputs = 0
    # if show_outputs:
    #     show_some_outputs(output_file)
    #     quit()

    # if it exists, remove file
    if overwrite and output_file.exists():
        output_file.unlink()
        print(f"Removed existing file {output_file}")


    # # Load dataset
    # if Path(dataset_name_or_path).exists():
    #     dataset = load_from_disk(dataset_name_or_path)
    # else:
    #     print ('\n loading dataset')
    #     dataset = load_dataset(dataset_name_or_path)
    # print ('dataset loaded \n')
    # dataset = dataset[split]



    # Load dataset
    arrow_path = "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-test.arrow"
    # arrow_path2 = "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/d335ae214fcf59e2f6530e5ea1f2ad67bb0c30ee/swe-bench_oracle-test.arrow"
    # load dataset
    dataset = load_dataset("arrow", data_files=arrow_path)["train"] #its actually test set.
    # print (dataset.column_names)
    # print (len(dataset))
    total_instances = len(dataset)
    print (f"Number of instances: {total_instances}")
    # convert to df
    # df = pd.DataFrame(dataset)
    # print




    # 
    # Sort by length
    lens = np.array(list(map(len, dataset["text"])))
    dataset = dataset.select(np.argsort(lens))

    print (dataset.column_names)
    print ()
    

    datapoints = filter_examples(dataset, output_file, max_length, past_output, past_logs)
    fasd


    # # SAVE INSTANCE IDS
    # instance_ids = [datum["instance_id"] for datum in datapoints]
    # print (instance_ids[:5])
    # instance_ids_output_path = "/home/chris_cohere_ai/SWE-bench-stuff/instance_ids/test_lite_cc_750.json"
    # # make dir if doesnt exist
    # os.makedirs(os.path.dirname(instance_ids_output_path), exist_ok=True)
    # with open(instance_ids_output_path, "w") as f:
    #     json.dump(instance_ids, f)
    # print (f"Saved to {instance_ids_output_path}")
    # fdsafs

    cohere_inference(
            datapoints=datapoints,
            model_path=model_path,
            output_file=output_file,
    )

    # if model_name_or_path.startswith("claude"):
    #     anthropic_inference(**inference_args)
    # elif model_name_or_path.startswith("gpt"):
    #     openai_inference(**inference_args)
    # else:
    #     raise ValueError(f"Invalid model name or path {model_name_or_path}")
    print(f"Done!")

