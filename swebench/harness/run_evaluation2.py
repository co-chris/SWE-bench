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
from swebench.harness.engine_evaluation import main as eval_engine
from swebench.harness.utils import get_instances
from swebench.metrics.getters import get_eval_refs

from swebench.harness.colours import blue

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation")


def deterministic_hash(input_string: str, length: int = None):
    input_bytes = input_string.encode('utf-8')
    sha256_hash = hashlib.sha256(input_bytes)
    hex_digest = sha256_hash.hexdigest()
    if length is None:
        return hex_digest
    return hex_digest[:length]


def validate_predictions(predictions_path, tasks_ids):
    # Check that predictions file exists
    if not any([predictions_path.endswith(x) for x in [".json", ".jsonl"]]):
        raise ValueError("Predictions path must be .json or .jsonl file")
    predictions = get_instances(predictions_path)
    not_in_tasks = []
    # Check that predictions are correctly formatted
    for pred in predictions:
        if any([x not in pred for x in [KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION]]):
            raise ValueError(f"Every prediction must have {KEY_INSTANCE_ID}, {KEY_MODEL}, and {KEY_PREDICTION} fields")
        if pred[KEY_INSTANCE_ID] not in tasks_ids:
            not_in_tasks.append(pred[KEY_INSTANCE_ID])
    # Check that instance IDs specified by predictions exist
    if len(not_in_tasks) > 0:
        logger.warning(
            "Predictions for the following instance_ids were not "
            + "found in the tasks file and will not be considered: "
            + ", ".join(not_in_tasks)
        )


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

    # Verify arguments are formatted correctly
    if not isinstance(tasks, list):
        raise ValueError(f"{swe_bench_tasks} must contain an array of tasks")
    tasks_map = {t[KEY_INSTANCE_ID]: t for t in tasks}
    predictions_path = os.path.abspath(predictions_path)
    validate_predictions(predictions_path, [t[KEY_INSTANCE_ID] for t in tasks])

    # Group predictions by model
    predictions = get_instances(predictions_path)
    map_model_to_predictions = {}
    for p in predictions:
        model = p[KEY_MODEL]
        if model not in map_model_to_predictions:
            map_model_to_predictions[model] = []
        map_model_to_predictions[model].append(p)
    # logger.info(f"Found {len(predictions)} predictions across {len(map_model_to_predictions)} model(s) in predictions file")

    # only one model at a time
    assert len(map_model_to_predictions) == 1, "Only one model at a time is supported"
    model_name = list(map_model_to_predictions.keys())[0]
    predictions = map_model_to_predictions[model_name]
    n_all_preds = len(predictions)

    # remove ones that are already done
    if skip_existing:
        predictions_todo = []
        for p in predictions:
            log_file = os.path.join(log_dir, p[KEY_MODEL], f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.eval.log")
            if not os.path.exists(log_file):
                predictions_todo.append(p)
        predictions = predictions_todo

    # For each model, split predictions by repo + save to folder
    # for model, predictions in map_model_to_predictions.items():
    # Group predictions by repository, version in order to install dependencies
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
    print (f"Model: {blue(model_name)}")
    print (f"# of preds total: {blue(n_all_preds)}")
    print (f"# of evals completed: {blue(n_all_preds-len(predictions))}")
    print (f"# of evals todo: {blue(len(predictions))}")
    print (f"# of repos: {blue(len(map_repo_version_to_predictions))}")
    print ("########################################")


    eval_args = []
    temp_dirs = []
    count = 0
    # For each model/repo/version, create testbed folder and save predictions
    # And prepare args for evaluation
    for repo in map_repo_version_to_predictions:
        for version in map_repo_version_to_predictions[repo]:
            # Create model/repo/version specific testbed folder
            testbed_model_name = model
            if len(testbed_model_name) > 50:
                # Hash model name for temp_dir path if too long
                # Issue: https://github.com/conda/conda/issues/12250
                testbed_model_name = deterministic_hash(testbed_model_name, 10)
            testbed_model_repo_version_dir = os.path.join(
                testbed, testbed_model_name, repo.rsplit('__', 1)[-1], version)
            os.makedirs(testbed_model_repo_version_dir, exist_ok=True)

            # Create predictions file for model/repo/version
            file_name = f"{model}_{repo}_{version}_{predictions_path.split('/')[-1]}"
            file_path = os.path.join(testbed_model_repo_version_dir, file_name)
            if file_path.endswith(".jsonl"):
                file_path = file_path[:-1]

            # Create evaluation args
            args = argparse.Namespace()
            args.log_dir = os.path.join(log_dir, model)
            args.log_suffix = log_suffix
            args.num_workers = 1
            args.predictions_path = file_path
            args.skip_existing = skip_existing
            args.temp_dir = testbed_model_repo_version_dir
            args.timeout = timeout
            args.verbose = verbose
            args.conda_link = conda_link
            args.count = count
            count += 1

            # Save predictions to file
            with open(file_path, "w") as f:
                json.dump(map_repo_version_to_predictions[repo][version], f, indent=4)

            eval_args.append(args)
            temp_dirs.append(testbed_model_repo_version_dir)



    
    if len(eval_args) == 0:
        logger.info("No predictions to evaluate")
        return

    


    # Run evaluation on each model/repo
    num_processes = min(len(eval_args), num_processes) if num_processes > 0 else len(eval_args)

    print ("########################################")
    print (f"# of eval_args: {blue(len(eval_args))}")
    print (f"# of num_processes: {blue(num_processes)}")
    print ("########################################")


    try:
        if num_processes == 1:
            for args in eval_args:
                eval_engine(args)
        else:
            pool = Pool(processes=num_processes)
            pool.map(eval_engine, eval_args)
            pool.close()
            pool.join()
    finally:
        # Clean up
        for temp_dir in temp_dirs:
            # Kill all processes that are using the temp directory
            subprocess.run(f"lsof +D {temp_dir} | awk 'NR>1 {{print $2}}' | xargs kill", shell=True, capture_output=True)
            # Remove temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file (must be .json)", required=True)
    parser.add_argument("--log_dir", type=str, help="Path to log directory", required=True)
    parser.add_argument("--swe_bench_tasks", type=str, help="Path to dataset file or HF datasets name", required=True)
    parser.add_argument("--testbed", type=str, help="Path to testbed directory", required=True)
    parser.add_argument("--conda_link", type=str, default=None, help="(Optional) URL to conda installation to use")
    parser.add_argument("--log_suffix", type=str, help="(Optional) Suffix to append to log file names", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="(Optional) Skip existing logs")
    parser.add_argument("--timeout", type=int, help="(Optional) Timeout in seconds (default: 900)", default=900)
    parser.add_argument("--verbose", action="store_true", help="(Optional) Verbose mode")
    parser.add_argument("--num_processes", type=int, help="(Optional) Number of processes to use.", default=-1)
    args = parser.parse_args()
    logger.propagate = args.verbose
    main(**vars(args))
