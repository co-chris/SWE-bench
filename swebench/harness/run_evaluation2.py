#!/usr/bin/env python3

"""Run evaluation"""
import argparse
import datasets
import hashlib
import json
import logging
import os
import shutil
import subprocess

from datasets import load_dataset
from multiprocessing import Pool
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
)
# from swebench.harness.engine_evaluation import main as eval_engine
from swebench.harness.utils import get_instances, validate_predictions, deterministic_hash
from swebench.metrics.getters import get_eval_refs

from swebench.harness.colours import blue
from swebench.harness.engine_evaluation import evaluate_predictions
from swebench.harness.engine_validation import setup_testbed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation")




def main(
    predictions_path: str,
    swe_bench_tasks: str,
    log_dir: str,
    testbed: str,
    conda_link: str,
    log_suffix: str,
    skip_existing: bool,
    timeout: int,
    verbose: bool,
    num_processes: int = -1,
):
    """
    Runs evaluation on predictions for each model/repo/version combination.

    Args:
        predictions_path (str): Path to the predictions file.
        swe_bench_tasks (str): Path to the SWE-bench tasks file OR HF dataset name.
        log_dir (str): Path to the directory where logs will be saved.
        testbed (str): Path to the directory where testbeds will be saved.
        skip_existing (bool): Whether to skip evaluations for predictions that already have logs.
        timeout (int): Timeout for each evaluation.
        verbose (bool): Whether to print verbose output.

    Raises:
        ValueError: If log_dir is not a directory, testbed is not a directory, or swe_bench_tasks does not exist.
    """
    # Validate arguments
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")
    if not os.path.exists(testbed) or not os.path.isdir(testbed):
        raise ValueError("--testbed must exist and point at a directory")

    tasks = list(get_eval_refs(swe_bench_tasks).values())
    log_suffix = log_suffix if log_suffix is not None else ""



    # model_name = predictions_path.split("/")[-1].split("__")[0]
    run_name = predictions_path.split("/")[-1].split("__")[0].split('.')[0]
    model_log_dir = os.path.join(log_dir, run_name+log_suffix)
    
    print (blue(run_name))

    allow_overwrite = True
    print (f"Log dir: {blue(model_log_dir)}")
    # if it exists, stop
    if os.path.exists(model_log_dir) and not allow_overwrite:
        logger.info(f"Log directory {model_log_dir} already exists")
        return

    # Create log, temp directories if they don't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        

    print ('Loading predictions')
    predictions = get_instances(predictions_path)
    

    # Only keep instances that gold patch solves AND are less than 40k tokens, 818 of them
    test_instances_path = "/home/chris_cohere_ai/SWE-bench-stuff/instance_ids/test_lite_cc_750.json"
    test_instances = json.load(open(test_instances_path))
    predictions_todo = []
    for p in predictions:
        if p["instance_id"] in test_instances:
            predictions_todo.append(p)
    predictions = predictions_todo
    print (f"Filtered to {blue(len(predictions))}: solved by gold and less than max_length tokens")
    n_all_preds = len(predictions)

    # add log file
    for p in predictions:
        # log_file = os.path.join(model_log_dir, f"{p[KEY_INSTANCE_ID]}.{p['model_name_or_path']}.log")
        log_file = os.path.join(model_log_dir, f"{p['instance_id']}.log")
        p["log_file"] = log_file

    # remove ones that are already done
    if skip_existing:
        predictions_todo = []
        for p in predictions:
            # log_file = os.path.join(model_log_dir, f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.log")
            if not os.path.exists(log_file):
                # add log_file to p
                # p["log_file"] = log_file
                predictions_todo.append(p)
        predictions = predictions_todo
    



    # fdsasfd
    # for debugging, only do 2
    # predictions = predictions[:5]



    # For each model, split predictions by repo + save to folder
    # for model, predictions in map_model_to_predictions.items():
    # Group predictions by repository, version in order to install dependencies
    tasks_map = {t[KEY_INSTANCE_ID]: t for t in tasks}
    map_repo_version_to_predictions = {}
    for p in predictions:
        instance_id = p[KEY_INSTANCE_ID]
        repo = instance_id.rsplit("-", 1)[0]
        if repo not in map_repo_version_to_predictions:
            map_repo_version_to_predictions[repo] = {}
        t = tasks_map[instance_id]
        p.update(t)
        version = t["version"]
        if version not in map_repo_version_to_predictions[repo]:
            map_repo_version_to_predictions[repo][version] = []
        map_repo_version_to_predictions[repo][version].append(p)

    print ("########################################")
    print (f"Run name: {blue(run_name)}")
    print (f"# of preds total: {blue(n_all_preds)}")
    print (f"# of evals completed: {blue(n_all_preds-len(predictions))}")
    print (f"# of evals todo: {blue(len(predictions))}")
    print ("########################################")



    eval_args = []
    # temp_dirs = []
    count = 0
    testbed_model_dir = os.path.join(testbed, run_name)
    # For each model/repo/version, create testbed folder and save predictions
    # And prepare args for evaluation
    for repo in map_repo_version_to_predictions:
        for version in map_repo_version_to_predictions[repo]:

            prediction_list = map_repo_version_to_predictions[repo][version]


            # Create model/repo/version specific testbed folder
            testbed_model_repo_version_dir = os.path.join(
                testbed_model_dir, repo.rsplit('__', 1)[-1], version)
            os.makedirs(testbed_model_repo_version_dir, exist_ok=True)

            # Create predictions file for model/repo/version in testbed folder
            file_name = f"{run_name}_{repo}_{version}_{predictions_path.split('/')[-1]}"
            testbed_file = os.path.join(testbed_model_repo_version_dir, file_name)
            if testbed_file.endswith(".jsonl"): # make it .json
                testbed_file = testbed_file[:-1]
            # i dont see why it needs to copy the predictions..wait its not. . .wiat it is
            # well ive added log file to preds, so i guess that is why

            # # Create evaluation args
            # args = argparse.Namespace()
            # # args.log_dir = os.path.join(log_dir, model)
            # # args.log_suffix = log_suffix
            # # args.log_file = todo
            # # args.num_workers = 1
            # args.predictions_path = testbed_file
            # # args.predictions_list = prediction_list
            # # args.skip_existing = skip_existing
            # args.temp_dir = testbed_model_repo_version_dir
            # args.timeout = timeout
            # args.verbose = verbose
            # args.conda_link = conda_link
            # args.count = count

            args = {
                "task_instances": prediction_list,
                "predictions_path": testbed_file,
                "temp_dir": testbed_model_repo_version_dir,
                "log_dir": model_log_dir,
                "timeout": timeout,
                "verbose": verbose,
                "conda_link": conda_link,
                "count": count,
                "func": evaluate_predictions,
            }



            count += 1

            # Save predictions to file
            with open(testbed_file, "w") as f:
                json.dump(prediction_list, f, indent=4)

            eval_args.append(args)
            # temp_dirs.append(testbed_model_repo_version_dir)



    
    if len(eval_args) == 0:
        logger.info("No predictions to evaluate")
        return

    


    # Run evaluation on each model/repo
    num_processes = min(len(eval_args), num_processes) if num_processes > 0 else len(eval_args)
    num_processes = min(num_processes, os.cpu_count()-4)

    print ("########################################")
    print (f"# of repos: {blue(len(map_repo_version_to_predictions))}")
    print (f"# of eval_args: {blue(len(eval_args))}")
    print (f"# of num_processes: {blue(num_processes)}")
    print ("########################################")



    # task = {
    #     "task_instances": predictions,
    #     "func": evaluate_predictions,
    #     **vars(args),
    # }

    # if args.num_workers == 1:
    # setup_testbed(task)

    try:
        # if num_processes == 1:
        #     for args in eval_args:
        #         eval_engine(args)
        # else:
        pool = Pool(processes=num_processes)
        # pool.map(eval_engine, eval_args)
        pool.map(setup_testbed, eval_args)
        pool.close()
        # we do pool.join() here to wait for all processes to finish? 
        pool.join()
    finally:
        # # Clean up
        # for temp_dir in temp_dirs:
        #     # Kill all processes that are using the temp directory
        #     subprocess.run(f"lsof +D {temp_dir} | awk 'NR>1 {{print $2}}' | xargs kill", shell=True, capture_output=True)
        #     # Remove temp directory
        #     shutil.rmtree(temp_dir, ignore_errors=True)

        # Delete testbed_model_dir
        # shutil.rmtree(testbed_model_dir, ignore_errors=True)
        print (model_log_dir)
        print ("done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file (must be .json)", required=True)
    # parser.add_argument("--log_dir", type=str, help="Path to log directory", required=True)
    # parser.add_argument("--swe_bench_tasks", type=str, help="Path to dataset file or HF datasets name", required=True)
    # parser.add_argument("--testbed", type=str, help="Path to testbed directory", required=True)
    parser.add_argument("--conda_link", type=str, default=None, help="(Optional) URL to conda installation to use")
    parser.add_argument("--log_suffix", type=str, help="(Optional) Suffix to append to log file names", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="(Optional) Skip existing logs")
    parser.add_argument("--timeout", type=int, help="(Optional) Timeout in seconds (default: 900)", default=900)
    parser.add_argument("--verbose", action="store_true", help="(Optional) Verbose mode")
    parser.add_argument("--num_processes", type=int, help="(Optional) Number of processes to use.", default=-1)
    args = parser.parse_args()

    # add value called testbed to args
    args.swe_bench_tasks = "princeton-nlp/SWE-bench_oracle"
    args.log_dir = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir"
    args.testbed = "/home/chris_cohere_ai/SWE-bench-stuff/testbed"

    logger.propagate = args.verbose
    main(**vars(args))
