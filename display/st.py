"""
streamlit run display/st.py --server.address=100.84.152.25  --server.port=8501
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

repos = df["repo"].unique()
# print (repos)
# dropdown to select the repo
repo = st.selectbox("Select the repo", repos)
df = df[df["repo"] == repo]
# print (df.columns)
# # first row instance_id
# print (df["PASS_TO_PASS"].values[0])
# print (df["FAIL_TO_PASS"].values[0])
# fsda

st.text(f"Number of instances: {len(df)}")

# dropdown to select the instance_id
instance_id = st.selectbox("Select the instance_id", df["instance_id"])
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






output_dir = "/home/chris_cohere_ai/SWE-bench-stuff/outputs"
# list all the files in the output directory
files = os.listdir(output_dir)
files = [f for f in files if dataset_name in f]
models = [f.split('__')[0] for f in files]
# print (files)









def get_model_report3(model, instance_id): #, predictions_path, swe_bench_tasks, ):
    """
    """
    
    # Get log file
    log_dir = "/home/chris_cohere_ai/SWE-bench-stuff/log_dir"
    log_path = os.path.join(log_dir, f"{model}/{instance_id}.{model}.eval.log")

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









with cols[1]:
    st.markdown("<h3>Output</h3>", unsafe_allow_html=True)
    # dropdown to select the model
    model = st.selectbox("Select the model", models)
    # get the model output
    model_output_path = os.path.join(output_dir, files[models.index(model)])
    predictions = get_instances(model_output_path)
    print (f"Number of predictions: {len(predictions)}")
    st.text(f"{len(predictions)}/{total_instances} model generated patches")

    # # get the model output for the selected instance_id
    # print (predictions[0].keys())
    for pred in predictions:
        if pred["instance_id"] == instance_id:
            prediction = pred
            break

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



    tests_PASS_TO_PASS = df[df["instance_id"] == instance_id]["PASS_TO_PASS"].values[0]
    tests_FAIL_TO_PASS = df[df["instance_id"] == instance_id]["FAIL_TO_PASS"].values[0]
    # json.loads(datum[key])
    tests_PASS_TO_PASS = json.loads(tests_PASS_TO_PASS)
    tests_FAIL_TO_PASS = json.loads(tests_FAIL_TO_PASS)
    # print (tests_PASS_TO_PASS)
    # print ()
    # print (tests_FAIL_TO_PASS)
    # st.text(f"Number of PASS_TO_PASS tests: {len(tests_PASS_TO_PASS)}")
    # st.text(f"Number of FAIL_TO_PASS tests: {len(tests_FAIL_TO_PASS)}")
    # use markdown and make the number blue


    # st.markdown(f"Number of PASS_TO_PASS tests: <span style='color:green'>{len(tests_PASS_TO_PASS)}</span>", unsafe_allow_html=True)
    # st.markdown(f"Number of FAIL_TO_PASS tests: <span style='color:green'>{len(tests_FAIL_TO_PASS)}</span>", unsafe_allow_html=True)

    log_content, result = get_model_report3(model=model,
                                    instance_id=instance_id,) #



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

    # st.text(f"PASS_TO_PASS: {pass_to_pass_success}/{pass_to_pass_total}")
    # for test in report["PASS_TO_PASS"]["success"]:
    #     st.markdown(f"<p>{pass_text} {test}</p>", unsafe_allow_html=True)
    # for test in report["PASS_TO_PASS"]["failure"]:
    #     st.markdown(f"<p>{fail_text} {test}</p>", unsafe_allow_html=True)
    # st.text(f"FAIL_TO_PASS: {fail_to_pass_success}/{fail_to_pass_total}")
    # for test in report["FAIL_TO_PASS"]["success"]:
    #     st.markdown(f"<p>{pass_text} {test}</p>", unsafe_allow_html=True)
    # for test in report["FAIL_TO_PASS"]["failure"]:
    #     st.markdown(f"<p>{fail_text} {test}</p>", unsafe_allow_html=True)

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






    st.markdown("<br><h4>Log File</h4>", unsafe_allow_html=True)
    st.text(log_content)
