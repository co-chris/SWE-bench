"""
python -m inference.show_inputs
"""


from datasets import load_dataset




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

for instance in dataset:
    print (instance.keys())
    print (instance['test_patch'])
    break