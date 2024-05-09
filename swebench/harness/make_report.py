"""
python -m swebench.harness.make_report

run eval: python -m swebench.harness.run_evaluation --predictions_path /home/chris_cohere_ai/SWE-bench-stuff/swe-bench-example-preds.json --swe_bench_tasks princeton-nlp/SWE-bench --log_dir /home/chris_cohere_ai/SWE-bench-stuff/log_dir --testbed /home/chris_cohere_ai/SWE-bench-stuff/testbed
"""


import os
import json
import argparse

from swebench.metrics.report import get_eval_report, get_resolution_status
from swebench.metrics.log_parsers import MAP_REPO_TO_PARSER
from swebench.metrics.getters import get_eval_refs
from swebench.metrics.constants import (
    # INSTALL_FAIL,
    # APPLY_PATCH_FAIL,
    # APPLY_PATCH_PASS,
    # RESET_FAILED,
    # TESTS_ERROR,
    # TESTS_TIMEOUT,
    ResolvedStatus
)

from swebench.harness.colours import blue



def get_log_path(log_dir, model_version, instance_id, model):
    log_path = os.path.join(log_dir, f"{model_version}/{instance_id}.{model_version}.eval.log")
    # check if it exists
    if not os.path.exists(log_path):
        # v2
        log_path = os.path.join(log_dir, model_version, f"{instance_id}.{model}.log")

        # if "provided_patch" in model_version:
        #     log_path = os.path.join(log_dir, model_version, f"{instance_id}.their_provided_patch.log")

    return log_path





# def get_model_report2(model, predictions_path, swe_bench_tasks, log_dir, log_suffix):
def get_model_report2(predictions_path, log_dir):
    """
    """


    # Evaluation Log Constants
    APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
    APPLY_PATCH_PASS = ">>>>> Applied Patch"
    INSTALL_FAIL = ">>>>> Init Failed"
    INSTALL_PASS = ">>>>> Init Succeeded"
    RESET_FAILED = ">>>>> Reset Failed"
    TESTS_ERROR = ">>>>> Tests Errored"
    TESTS_TIMEOUT = ">>>>> Tests Timed Out"



    # instance_id: datapoint
    swe_bench_tasks = "princeton-nlp/SWE-bench_oracle"
    eval_refs = get_eval_refs(swe_bench_tasks)
    # print (eval_refs.keys())
    # fdsafa
    for k, v in eval_refs.items():
        eval_refs[k] = {key: v[key] for key in ["instance_id", "FAIL_TO_PASS", "PASS_TO_PASS"]}


    # Get predictions
    if predictions_path.endswith(".json"):
        predictions = json.load(open(predictions_path))
    else:
        # load jsonl
        predictions = []
        with open(predictions_path) as f:
            for line in f:
                predictions.append(json.loads(line))

    # Only keep instances that gold patch solves AND are less than 40k tokens, 818 of them
    test_instances_path = "/home/chris_cohere_ai/SWE-bench-stuff/instance_ids/test_lite_cc_750.json"
    test_instances = json.load(open(test_instances_path))
    predictions_todo = []
    for p in predictions:
        if p["instance_id"] in test_instances:
            predictions_todo.append(p)
    predictions = predictions_todo
    print (f"Filtered to {blue(len(predictions))}: solved by gold and less than max_length tokens")
    # n_all_preds = len(predictions)


    
    keys = ["model_patch_does_not_exist", "model_patch_exists", "with_logs", 
            "install_fail", 
            "APPLY_PATCH_FAIL",
            "RESET_FAILED",
            "TESTS_ERROR",
            "TESTS_TIMEOUT",
            "Failed to reset task environment",
            "applied", 
            "resolved"]
    report_map = {k: [] for k in keys}
    # Iterate through predictions'
    print (f"Total predictions: {len(predictions)}")
    for p in predictions:


        instance_id = p["instance_id"]
        # model_name = p["model_name_or_path"]
        # print (p.keys())
        # fadfasd
        # # fasd
        # 
        # if repo not in report_map:
        #     report_map[repo] = {k: [] for k in keys}

        # Check if the model patch exists
        if p["model_patch"] == None:
            report_map["model_patch_does_not_exist"].append(instance_id)
            continue
        report_map["model_patch_exists"].append(instance_id)

        # Get log file
        # log_path = os.path.join(log_dir, f"{model}/{instance_id}.{model}.eval.log")
        
        # if log_suffix is None:
        #     log_suffix = ''
        
        # v2
        # log_path = os.path.join(log_dir, model+log_suffix, f"{instance_id}.{model}.log")
        # log_path = get_log_path(log_dir, model+log_suffix, instance_id, model)
        log_path = os.path.join(log_dir, instance_id+'.log')


        # print (log_path)
        # fadsfa

        # print (log_path)
        # fasdf
        if not os.path.exists(log_path):
            print (f"#########Log file {log_path} does not exist###########")
            continue
        report_map["with_logs"].append(instance_id)

        # Check if the model patch was applied successfully
        with open(log_path) as f:
            log_content = f.read()

        # Check if install succeeded
        if INSTALL_FAIL in log_content:
            report_map["install_fail"].append(instance_id)
            continue
        # if any([x in content for x in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT, "Failed to reset task environment"]]) or APPLY_PATCH_PASS not in content:
        elif APPLY_PATCH_FAIL in log_content:
            # print (f"Failed to apply patch: {log_path}")
            report_map["APPLY_PATCH_FAIL"].append(instance_id)
            continue
        elif RESET_FAILED in log_content:
            # print (f"Failed to reset task environment: {log_path}")
            report_map["RESET_FAILED"].append(instance_id)
            continue
        elif TESTS_ERROR in log_content:
            # print (f"Tests errored out: {log_path}")
            report_map["TESTS_ERROR"].append(instance_id)
            continue
        elif TESTS_TIMEOUT in log_content:
            # print (f"Tests timed out: {log_path}")
            report_map["TESTS_TIMEOUT"].append(instance_id)
            continue
        elif "Failed to reset task environment" in log_content:
            # print (f"Failed to reset task environment: {log_path}")
            report_map["Failed to reset task environment"].append(instance_id)
            continue
            
            # # Eval patch was not applied successfully
            # return {}, False
        report_map["applied"].append(instance_id)

        # print (log_content)
        # print (instance_id)


        # Get status map of evaluation results
        passed_content = log_content.split(f"{APPLY_PATCH_PASS} (pred)")[-1]
        repo = instance_id.split(".")[0].rsplit("-", 1)[0].replace("__", "/")
        log_parser = MAP_REPO_TO_PARSER[repo]
        tests_statuses = log_parser(passed_content)
        print (blue(tests_statuses))
        # fasdf
        expected_statuses = eval_refs[instance_id]
        print ('expected_statuses')
        print (expected_statuses)
        # continue

        report = get_eval_report(tests_statuses, expected_statuses)
        pass_to_pass_success = len(report["PASS_TO_PASS"]["success"])
        pass_to_pass_total = len(report["PASS_TO_PASS"]["success"]) + len(report["PASS_TO_PASS"]["failure"])
        fail_to_pass_success = len(report["FAIL_TO_PASS"]["success"])
        fail_to_pass_total = len(report["FAIL_TO_PASS"]["success"]) + len(report["FAIL_TO_PASS"]["failure"])
        
        # print (report["PASS_TO_PASS"]["success"])
        # print (report["FAIL_TO_PASS"]["success"])
        # print (report["PASS_TO_PASS"]["failure"])
        # print (report["FAIL_TO_PASS"]["failure"])
        # fads
        
        
        
        # print (f"{instance_id}: {pass_to_pass_success}/{pass_to_pass_total} {fail_to_pass_success}/{fail_to_pass_total}")
        # # print (report)
        # for k, v in report.items():
        #     print (f"{k}: {v}")
        # fdsafa

        if get_resolution_status(report) == ResolvedStatus.FULL.value:
            report_map["resolved"].append(instance_id)
            # print (f"Resolved: {instance_id}")
        else:
            print (blue('-------------------------'))
            print (log_content)
            print (blue('-------------------------'))

            print (report["PASS_TO_PASS"]["success"])
            print (report["FAIL_TO_PASS"]["success"])
            print (report["PASS_TO_PASS"]["failure"])
            print (report["FAIL_TO_PASS"]["failure"])
            fads

    return report_map












if __name__ == "__main__":

    # model = "test"
    # predictions_path = "/home/chris_cohere_ai/SWE-bench-stuff/swe-bench-example-preds.json"
    # # swe_bench_tasks = "princeton-nlp/SWE-bench"
    # swe_bench_tasks = "/home/chris_cohere_ai/SWE-bench-stuff/tasks/test_set/swe-bench.json"
    # log_dir = f"/home/chris_cohere_ai/SWE-bench-stuff/log_dir/{model}"


    # model = "their_provided_patch"
    # predictions_path = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/provided_patch.json"
    # swe_bench_tasks = "princeton-nlp/SWE-bench_oracle"
    # log_dir = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir"

    # # python -m swebench.harness.run_evaluation2 --predictions_path /home/chris_cohere_ai/SWE-bench-stuff/outputs/command-r__SWE-bench_oracle__test.jsonl --swe_bench_tasks princeton-nlp/SWE-bench_oracle --log_dir /home/chris_cohere_ai/SWE-bench-stuff/log_dir --testbed /home/chris_cohere_ai/SWE-bench-stuff/testbed --timeout=60 --skip_existing
    # model = "command-r"
    # predictions_path = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/command-r__SWE-bench_oracle__test.jsonl"
    # swe_bench_tasks = "princeton-nlp/SWE-bench_oracle"
    # log_dir = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir"

    # model = "command-r-plus"
    # predictions_path = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/command-r-plus__SWE-bench_oracle__test.jsonl"
    # swe_bench_tasks = "princeton-nlp/SWE-bench_oracle"
    # log_dir = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir"

    # model = "their_provided_patch"
    # log_suffix = '_2'
    # predictions_path = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/provided_patch.json"
    # swe_bench_tasks = "princeton-nlp/SWE-bench_oracle"
    # log_dir = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir"

    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file (must be .json)", required=True)
    # parser.add_argument("--log_dir", type=str, help="Path to log directory", required=True)
    # parser.add_argument("--swe_bench_tasks", type=str, help="Path to dataset file or HF datasets name", required=True)
    # parser.add_argument("--log_suffix", type=str, help="(Optional) Suffix to append to log file names", default=None)
    args = parser.parse_args()

    log_dir = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir/provided_patch_2024_05_09"

    # report = get_model_report2(args.model, args.predictions_path, args.swe_bench_tasks, args.log_dir, args.log_suffix)
    report = get_model_report2(args.predictions_path, log_dir)
    keys = list(report.keys())
    for k in keys:
        print (f"{k}: {len(report[k])}")
    # repos = [k for k in report.keys()]
    # keys = []
    # for k, v in report.items():
    #     keys.extend(v.keys())

    # # print (f"Repos: {len(repos)}")
    # print ("")
    # for key in keys:
    #     print (f"\t{key}: {sum([len(v[key]) for k, v in report.items() if isinstance(v, dict)])}")



    # print the instance ID based on how far it gets through the eval
    print (f"\nInstances:")
    hierchy_dict = {k: [] for k in keys}
    # task_name = 'django/django'
    # print (report.keys())
    # repo_names= list(report.keys())
    # fasdf
    # get all instances for a task
    all_instances = []
    for status, instances in report.items():
        all_instances.extend(instances)
    all_instances = set(all_instances)
    # go through the hierchy in reverse order and add to dict if it exists
    for instance in all_instances:
        for status in keys[::-1]:
            if instance in report[status]:
                hierchy_dict[status].append(instance)
                break
    # print the instances in order
    for status in keys:
        if len(hierchy_dict[status]) == 0:
            continue
        print (f"{status}: {len(hierchy_dict[status])} instances")
        # for instance in hierchy_dict[status]:
        #     print (f"\t{instance}")

print ()


# # print all the resovled instances
# print ("Resolved Instances:")
# for instance in report["resolved"]:
#     print (f"\t{instance}")

# output_path = "/home/chris_cohere_ai/SWE-bench-stuff/tasks/resolved_instances/swe-bench-oracle-resolved.json"
# with open(output_path, 'w') as f:
#     json.dump(report["resolved"], f, indent=4)
# print (f"Saved resolved instances to {output_path}")

