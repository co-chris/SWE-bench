"""
streamlit run display/st2.py --server.address=100.84.152.25  --server.port=8501
"""

import streamlit as st
import streamlit.components.v1 as components
from datasets import load_dataset
import os
import json
import pandas as pd

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import DiffLexer

# from swebench.harness.utils import get_instances

# ImportError: attempted relative import with no known parent package
# from ..harness.utils import get_instances

# ImportError: attempted relative import with no known parent package
# from swebench.harness.utils import get_instances
# ModuleNotFoundError: No module named 'swebench'
# from harness.utils import get_instances
# ModuleNotFoundError: No module named 'harness'
# from utils import get_instances

import sys
sys.path.insert(0, '.')
from swebench.harness.utils import get_instances
from swebench.metrics.log_parsers import MAP_REPO_TO_PARSER
from swebench.metrics.report import get_eval_report
# from swebench.harness.make_report import get_model_report2

lexer = DiffLexer()
formatter = HtmlFormatter()

def highlight_patch(patch):
    formatted_diff = highlight(patch, lexer, formatter)
    css = """
    <head>
        <style>
            .gd {
                background-color: #f8cbad;
            }
            .gi {
                background-color: #d9ead3;
            }
            .gu {
                background-color: #bdc4ba;
            }
            body {
                background-color: #eae5e1;
            }

        </style>
    </head>
    """
    formatted_diff = f"<html>{css}<body>{formatted_diff}</body></html>"
    return formatted_diff






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


def display_applied_content(log_content, instance_id, tests_PASS_TO_PASS, tests_FAIL_TO_PASS):

    report = get_tests_results(log_content, instance_id, tests_PASS_TO_PASS, tests_FAIL_TO_PASS)
    pass_to_pass_success = len(report["PASS_TO_PASS"]["success"])
    pass_to_pass_total = len(report["PASS_TO_PASS"]["success"]) + len(report["PASS_TO_PASS"]["failure"])
    fail_to_pass_success = len(report["FAIL_TO_PASS"]["success"])
    fail_to_pass_total = len(report["FAIL_TO_PASS"]["success"]) + len(report["FAIL_TO_PASS"]["failure"])
    
    # print (report["PASS_TO_PASS"]["success"])
    # print (report["FAIL_TO_PASS"]["success"])
    # print (report["PASS_TO_PASS"]["failure"])
    # print (report["FAIL_TO_PASS"]["failure"])


    # st.markdown("<h3>Results</h3>", unsafe_allow_html=True)
    # st.text(result)

    pass_text = "<span style='color:green'>PASSED</span>"
    fail_text = "<span style='color:red'>FAILED</span>"


    # too much spacing between tests, so reduce the spacing between <p> tags
    st.text(f"PASS_TO_PASS: {pass_to_pass_success}/{pass_to_pass_total}")
    for test in report["PASS_TO_PASS"]["success"]:
        st.markdown(f"<p style='margin: 0px'>{pass_text} {test}</p>", unsafe_allow_html=True)
    for test in report["PASS_TO_PASS"]["failure"]:
        st.markdown(f"<p style='margin: 0px'>{fail_text} {test}</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.text(f"FAIL_TO_PASS: {fail_to_pass_success}/{fail_to_pass_total}")
    for test in report["FAIL_TO_PASS"]["success"]:
        st.markdown(f"<p style='margin: 0px'>{pass_text} {test}</p>", unsafe_allow_html=True)
    for test in report["FAIL_TO_PASS"]["failure"]:
        st.markdown(f"<p style='margin: 0px'>{fail_text} {test}</p>", unsafe_allow_html=True)


def get_log_path(log_dir, model_version, instance_id, model):
    log_path = os.path.join(log_dir, f"{model_version}/{instance_id}.{model_version}.eval.log")
    # check if it exists
    if not os.path.exists(log_path):
        # v2
        log_path = os.path.join(log_dir, model_version, f"{instance_id}.{model}.log")

        # if "provided_patch" in model_version:
        #     log_path = os.path.join(log_dir, model_version, f"{instance_id}.their_provided_patch.log")

    return log_path















# data_info_path = "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/dataset_info.json"
# with open(data_info_path, "r") as file:
#     data_info = json.load(file)
#     print (data_info)
print ('-------------------------')
dataset_name = 'SWE-bench_oracle'
arrow_path = "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/b1f5f0b261409a0df9ac19f10bd07b88a8d9d4a2/swe-bench_oracle-test.arrow"
# arrow_path2 = "/home/chris_cohere_ai/.cache/huggingface/datasets/princeton-nlp___swe-bench_oracle/default/0.0.0/d335ae214fcf59e2f6530e5ea1f2ad67bb0c30ee/swe-bench_oracle-test.arrow"
# load dataset
dataset = load_dataset("arrow", data_files=arrow_path)["train"]
# print (dataset.column_names)
# print (len(dataset))
total_instances = len(dataset)
print (f"Number of instances: {total_instances}")
# convert to df
df = pd.DataFrame(dataset)
# print (df.head())

# dataset = load_dataset("arrow", data_files=arrow_path2)["train"]
# print (dataset.column_names)
# print (len(dataset))
# fasd

st.set_page_config(page_title="swe bench", layout="wide", page_icon="🎯")
















st.markdown("<center><h1>SWE Bench Viewer</h1></center>", unsafe_allow_html=True)

# dataset_name_or_path = "princeton-nlp/SWE-bench_oracle"
# dataset = load_dataset(dataset_name_or_path, split="test")
# print (dataset.column_names)
# print (len(dataset))
# fsafd

output_dir = "/home/chris_cohere_ai/SWE-bench-stuff/outputs"
log_dir = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir"


# list all the files in the output directory
output_files = os.listdir(output_dir)
output_files = [f for f in output_files if dataset_name in f]
models = [f.split('__')[0] for f in output_files]

# print (output_files)
# print (models)
# files.append("provided_patch.json")
# models.append("provided_patch")


cols = st.columns(2)

with cols[0]:
    # dropdown to select the model
    model = st.selectbox("Select model", models)


# # find suffixes
# log_dirs_models = os.listdir(log_dir)
# log_dirs_models = [f for f in log_dirs_models if model in f]
# if len(log_dirs_models) > 1 and 'command-r' not in log_dirs_models:
#     with cols[1]:
#         model_version = st.selectbox("Select version", log_dirs_models)
# else:
#     model_version = model

# print (model_version)

model_version = model


# get the model output
model_output_path = os.path.join(output_dir, output_files[models.index(model)])
# print (model_output_path)
predictions = get_instances(model_output_path)

# # if pytest not in instance id, remove it
# predictions = [p for p in predictions if "pytest-dev__pytest-5227" in p["instance_id"]]


# Group the instances
# - not generated
# - not applied
# - fails tests
# - resolves tests


# check if results file exists
results_path = os.path.join(log_dir, model_version, "results.json")
if os.path.exists(results_path):
    groups = json.load(open(results_path))

else:
    groups = {
        # "not_generated": [],
        "resolves_tests": [],
        "fails_tests": [],
        "patch_applied_failed": [],
    }

    for pred in predictions:
        instance_id = pred["instance_id"]
        tests_PASS_TO_PASS = json.loads(df[df["instance_id"] == instance_id]["PASS_TO_PASS"].values[0])
        tests_FAIL_TO_PASS = json.loads(df[df["instance_id"] == instance_id]["FAIL_TO_PASS"].values[0])

        log_path = get_log_path(log_dir, model_version, instance_id, model)
        # log_path = os.path.join(log_dir, "provided_patch", f"{instance_id}.their_provided_patch.eval.log")
        # print (log_path)
        log_content, result = get_model_report3(log_path=log_path)

        # print (pred.keys()) 
        
        if ">>>>> Applied Patch (pred)" not in log_content and ">>>>> Applied Patch (PatchType.PATCH_PRED)" not in log_content:
            groups["patch_applied_failed"].append(instance_id)
        else:
            report = get_tests_results(log_content, instance_id, tests_PASS_TO_PASS, tests_FAIL_TO_PASS)
            if len(report["PASS_TO_PASS"]["failure"]) == 0 and len(report["FAIL_TO_PASS"]["failure"]) == 0:
                groups["resolves_tests"].append(instance_id)
            else:
                groups["fails_tests"].append(instance_id)

    # if group is empty, remove it
    groups = {k: v for k, v in groups.items() if v}

    # save the groups
    with open(results_path, "w") as file:
        json.dump(groups, file)
        print (f"Results saved to {results_path}")








with cols[0]:
    print (f"Number of predictions: {len(predictions)}")
    st.text(f"{len(predictions)}/{total_instances} model generated patches")
    # Display counts of each group
    for group, instances in groups.items():
        st.text(f"{group}: {len(instances)}")

with cols[0]:
    # Select the group
    group = st.selectbox("Select task status", list(groups.keys()))
    # Select the instance_id
    instance_id = st.selectbox("Select instance_id", groups[group])









# fasdfsd


# repos = df["repo"].unique()
# # print (repos)
# # dropdown to select the repo
# repo = st.selectbox("Select repo", repos)
# df = df[df["repo"] == repo]
# # print (df.columns)
# # # first row instance_id
# # print (df["PASS_TO_PASS"].values[0])
# # print (df["FAIL_TO_PASS"].values[0])
# # fsda

# st.text(f"Number of instances: {len(df)}")

# # dropdown to select the instance_id
# instance_id = st.selectbox("Select instance_id", df["instance_id"])




input_text = df[df["instance_id"] == instance_id]["text"].values[0]
input_text_tokens = len(input_text) / 3.4
# split on "</issue>\n<code>"
split_input_text = input_text.split("</issue>\n<code>")
input_text1 = split_input_text[0] + "</issue>\n"
input_text2 = "<code>" + split_input_text[1]
split_input_text2 = input_text2.split("</code>")
input_text2 = split_input_text2[0] + "</code>"
input_text2_patch = split_input_text2[1]




patch = df[df["instance_id"] == instance_id]["patch"].values[0]
# patch = df[df["instance_id"] == instance_id]["test_patch"].values[0]
tests_PASS_TO_PASS = df[df["instance_id"] == instance_id]["PASS_TO_PASS"].values[0]
tests_FAIL_TO_PASS = df[df["instance_id"] == instance_id]["FAIL_TO_PASS"].values[0]
tests_PASS_TO_PASS = json.loads(tests_PASS_TO_PASS)
tests_FAIL_TO_PASS = json.loads(tests_FAIL_TO_PASS)

















cols = st.columns(2)
with cols[0]:
    st.markdown("<h3>Input</h3>", unsafe_allow_html=True)
    # st.code(input_text, language="python")
    # click to show code then it drops down
    # st.markdown(f"""<details>
    #             <summary>Click to show prompt</summary>
    #             {input_text1}
    #             </details>""", unsafe_allow_html=True)  
    st.text(f"Number of tokens: {int(input_text_tokens):,}")
    with st.expander("Issue", expanded=False):
        st.markdown(input_text1, unsafe_allow_html=True)
    with st.expander("Code", expanded=False):
        st.code(input_text2, language="python")
    with st.expander("Patch Prompt", expanded=False):
        st.code(input_text2_patch, language="python")




    st.markdown("<h3>Gold Patch</h3>", unsafe_allow_html=True)
    # html_to_display_path = "/home/chris_cohere_ai/SWE-bench-fork/display/diff.html"
    # display the diff content
    # with open(html_to_display_path, "r") as file:
    #     html_to_display = file.read()
    #     # components.html(html_to_display, scrolling=True)
    #     components.html(html_to_display, width=800, scrolling=True)

    patch = patch.replace("<patch>", "")
    patch = patch.replace("</patch>", "")
    formatted_diff = highlight_patch(patch)
    components.html(formatted_diff, scrolling=True, height=500)

    # st.markdown("<h3>Gold Patch</h3>", unsafe_allow_html=True)
    # st.code(patch, language="diff")

    # gold_log_path = os.path.join(log_dir, "provided_path", f"{instance_id}.their_provided_patch.log")
    gold_log_path = os.path.join(log_dir, "provided_patch", f"{instance_id}.their_provided_patch.eval.log")
    gold_log_content, gold_result = get_model_report3(log_path=gold_log_path)
    # print (gold_log_path)

    if ">>>>> Applied Patch (pred)" in gold_log_content:
        display_applied_content(gold_log_content, instance_id, tests_PASS_TO_PASS, tests_FAIL_TO_PASS)
    else:
        st.text("Patch not applied")

    st.markdown("<br><h4>Log File</h4>", unsafe_allow_html=True)
    st.text(gold_log_content)































with cols[1]:
    st.markdown("<h3>Output</h3>", unsafe_allow_html=True)

    # log_path = os.path.join(log_dir, f"{model_version}/{instance_id}.{model_version}.eval.log")
    # # check if it exists
    # if not os.path.exists(log_path):
    #     # v2
    #     log_path = os.path.join(log_dir, model_version, f"{instance_id}.{model}.log")

    #     if "provided_patch" in model_version:
    #         log_path = os.path.join(log_dir, model_version, f"{instance_id}.their_provided_patch.log")

    log_path = get_log_path(log_dir, model_version, instance_id, model)

    # # get the model output for the selected instance_id
    # print (predictions[0].keys())
    prediction = None
    for pred in predictions:
        if pred["instance_id"] == instance_id:
            prediction = pred
            break
    if prediction is None:
        st.text("No prediction found")
        st.stop()

    if "full_output" in prediction:

        output_text_tokens = len(prediction['full_output']) / 3.4
        st.text(f"Number of tokens: {int(output_text_tokens):,}")

        with st.expander("Full Output", expanded=False):
            # st.markdown(f"{prediction['full_output']}", unsafe_allow_html=True)
            st.code(f"{prediction['full_output']}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Model Generated Patch</h3>", unsafe_allow_html=True)
    # with st.expander("Patch", expanded=False):
    formatted_diff = highlight_patch(prediction["model_patch"])
    # st.code(formatted_diff, language="diff")
    components.html(formatted_diff, scrolling=True, height=500)



    st.markdown("<h3>Tests</h3>", unsafe_allow_html=True)
    log_content, result = get_model_report3(log_path=log_path)

    if ">>>>> Applied Patch (pred)" not in log_content and ">>>>> Applied Patch (PatchType.PATCH_PRED)" not in log_content:
        st.text("Patch not applied")
    else:
        display_applied_content(log_content, instance_id, tests_PASS_TO_PASS, tests_FAIL_TO_PASS)
        

    st.markdown("<br><h4>Log File</h4>", unsafe_allow_html=True)
    st.text(log_content)
