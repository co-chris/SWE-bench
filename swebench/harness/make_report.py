"""
python -m swebench.harness.make_report

run eval: python -m swebench.harness.run_evaluation --predictions_path /home/chris_cohere_ai/SWE-bench-stuff/swe-bench-example-preds.json --swe_bench_tasks princeton-nlp/SWE-bench --log_dir /home/chris_cohere_ai/SWE-bench-stuff/log_dir --testbed /home/chris_cohere_ai/SWE-bench-stuff/testbed
"""


import os
import json

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



# Evaluation Log Constants
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"
INSTALL_FAIL = ">>>>> Init Failed"
INSTALL_PASS = ">>>>> Init Succeeded"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"






def get_model_report2(model, predictions_path, swe_bench_tasks, log_dir):
    """
    """

    eval_refs = get_eval_refs(swe_bench_tasks)
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
        # fasd
        # 
        # if repo not in report_map:
        #     report_map[repo] = {k: [] for k in keys}

        # Check if the model patch exists
        if p["model_patch"] == None:
            report_map["model_patch_does_not_exist"].append(instance_id)
            continue
        report_map["model_patch_exists"].append(instance_id)

        # Get log file
        log_path = os.path.join(log_dir, f"{model}/{instance_id}.{model}.eval.log")
        # print (log_path)
        # fasdf
        if not os.path.exists(log_path):
            # print (f"#########Log file {log_path} does not exist###########")
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

        # Get status map of evaluation results
        passed_content = log_content.split(f"{APPLY_PATCH_PASS} (pred)")[-1]
        repo = instance_id.split(".")[0].rsplit("-", 1)[0].replace("__", "/")
        tests_statuses = MAP_REPO_TO_PARSER[repo](passed_content)
        expected_statuses = eval_refs[instance_id]

        report = get_eval_report(tests_statuses, expected_statuses)
        pass_to_pass_success = len(report["PASS_TO_PASS"]["success"])
        pass_to_pass_total = len(report["PASS_TO_PASS"]["success"]) + len(report["PASS_TO_PASS"]["failure"])
        fail_to_pass_success = len(report["FAIL_TO_PASS"]["success"])
        fail_to_pass_total = len(report["FAIL_TO_PASS"]["success"]) + len(report["FAIL_TO_PASS"]["failure"])
        # print (f"{instance_id}: {pass_to_pass_success}/{pass_to_pass_total} {fail_to_pass_success}/{fail_to_pass_total}")
        # # print (report)
        # for k, v in report.items():
        #     print (f"{k}: {v}")
        # fdsafa

        if get_resolution_status(report) == ResolvedStatus.FULL.value:
            report_map["resolved"].append(instance_id)

    return report_map












if __name__ == "__main__":

    # model = "test"
    # predictions_path = "/home/chris_cohere_ai/SWE-bench-stuff/swe-bench-example-preds.json"
    # # swe_bench_tasks = "princeton-nlp/SWE-bench"
    # swe_bench_tasks = "/home/chris_cohere_ai/SWE-bench-stuff/tasks/test_set/swe-bench.json"
    # log_dir = f"/home/chris_cohere_ai/SWE-bench-stuff/log_dir/{model}"


    model = "their_provided_patch"
    predictions_path = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/provided_patch.json"
    swe_bench_tasks = "princeton-nlp/SWE-bench_oracle"
    log_dir = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir"

    # # python -m swebench.harness.run_evaluation2 --predictions_path /home/chris_cohere_ai/SWE-bench-stuff/outputs/command-r__SWE-bench_oracle__test.jsonl --swe_bench_tasks princeton-nlp/SWE-bench_oracle --log_dir /home/chris_cohere_ai/SWE-bench-stuff/log_dir --testbed /home/chris_cohere_ai/SWE-bench-stuff/testbed --timeout=60 --skip_existing
    # model = "command-r"
    # predictions_path = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/command-r__SWE-bench_oracle__test.jsonl"
    # swe_bench_tasks = "princeton-nlp/SWE-bench_oracle"
    # log_dir = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir"



    report = get_model_report2(model, predictions_path, swe_bench_tasks, log_dir)
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
