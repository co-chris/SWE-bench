import os

from swebench.metrics.log_parsers import MAP_REPO_TO_PARSER
from swebench.metrics.report import get_eval_report


def get_log_path(log_dir, model_version, instance_id, model):
    log_path = os.path.join(log_dir, f"{model_version}/{instance_id}.{model_version}.eval.log")
    # check if it exists
    if not os.path.exists(log_path):
        # v2
        log_path = os.path.join(log_dir, model_version, f"{instance_id}.{model}.log")

        # if "provided_patch" in model_version:
        #     log_path = os.path.join(log_dir, model_version, f"{instance_id}.their_provided_patch.log")

    return log_path



def get_model_report3(log_path): #, predictions_path, swe_bench_tasks, ):

    # Evaluation Log Constants
    APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
    # APPLY_PATCH_PASS = ">>>>> Applied Patch"
    INSTALL_FAIL = ">>>>> Init Failed"
    # INSTALL_PASS = ">>>>> Init Succeeded"
    RESET_FAILED = ">>>>> Reset Failed"
    TESTS_ERROR = ">>>>> Tests Errored"
    TESTS_TIMEOUT = ">>>>> Tests Timed Out"


    if not os.path.exists(log_path):
        return 'no log file', 'no log file'

    # Check if the model patch was applied successfully
    with open(log_path) as f:
        log_content = f.read()
        # return log_content, 'applied maybe'

    # Check if install succeeded
    if INSTALL_FAIL in log_content:
        return log_content, "Install failed"
    elif APPLY_PATCH_FAIL in log_content:
        return log_content, "Patch apply failed"
    elif RESET_FAILED in log_content:
        return log_content, "Reset failed"
    elif TESTS_ERROR in log_content:
        return log_content, "Tests error"
    elif TESTS_TIMEOUT in log_content:
        return log_content, "Tests timeout"
    elif "Failed to reset task environment" in log_content:
        return log_content, "Failed to reset task environment"
    
    return log_content, "applied but not sure if tests passed"
        
    #     # # Eval patch was not applied successfully
    #     # return {}, False
    # report_map["applied"].append(instance_id)

    # # Get status map of evaluation results
    # passed_content = log_content.split(f"{APPLY_PATCH_PASS} (pred)")[-1]
    # repo = instance_id.split(".")[0].rsplit("-", 1)[0].replace("__", "/")
    # tests_statuses = MAP_REPO_TO_PARSER[repo](passed_content)
    # expected_statuses = eval_refs[instance_id]

    # report = get_eval_report(tests_statuses, expected_statuses)
    # pass_to_pass_success = len(report["PASS_TO_PASS"]["success"])
    # pass_to_pass_total = len(report["PASS_TO_PASS"]["success"]) + len(report["PASS_TO_PASS"]["failure"])
    # fail_to_pass_success = len(report["FAIL_TO_PASS"]["success"])
    # fail_to_pass_total = len(report["FAIL_TO_PASS"]["success"]) + len(report["FAIL_TO_PASS"]["failure"])
    # # print (f"{instance_id}: {pass_to_pass_success}/{pass_to_pass_total} {fail_to_pass_success}/{fail_to_pass_total}")
    # # # print (report)
    # # for k, v in report.items():
    # #     print (f"{k}: {v}")
    # # fdsafa

    # if get_resolution_status(report) == ResolvedStatus.FULL.value:
    #     report_map["resolved"].append(instance_id)

    # return report_map

def get_tests_results(log_content, instance_id, tests_PASS_TO_PASS, tests_FAIL_TO_PASS):
    # Get status map of evaluation results
    passed_content = log_content.split(f">>>>> Applied Patch (pred)")[-1]
    repo = instance_id.split(".")[0].rsplit("-", 1)[0].replace("__", "/")
    log_parser = MAP_REPO_TO_PARSER[repo]
    tests_statuses = log_parser(passed_content)
    # expected_statuses = eval_refs[instance_id]
    expected_statuses = {
        "instance_id": instance_id,
        "PASS_TO_PASS": tests_PASS_TO_PASS,
        "FAIL_TO_PASS": tests_FAIL_TO_PASS,
    }
    # print (expected_statuses)

    report = get_eval_report(tests_statuses, expected_statuses)
    return report

