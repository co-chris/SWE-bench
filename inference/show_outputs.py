"""
python -m inference.show_outputs
"""

import os
import json
from swebench.harness.colours import blue

from swebench.harness.utils import extract_minimal_patch

from inference.make_datasets.utils import extract_diff


def show_some_outputs(output_file):
    # Load output file
    # if output_file.exists():
    # if os.path.exists(output_file):
    patch_count = 0
    patch_end_count = 0
    mycheck = 0
    repetition = 0
    total = 0
    rep_counts = []
    with open(output_file, "r") as f:
        for i, line in enumerate(f):

            show_it = 0

            # if i !=7:
            #     continue

            data = json.loads(line)
            print (data.keys())
            fdsaf
            full_output = data['full_output']

            total += 1
            if "<patch>" in full_output:
                patch_count += 1
            if "</patch>" in full_output:
                patch_end_count += 1

            if '@@' in full_output and '---' in full_output:
                mycheck +=1
            # else:
            #     show_it = 1

            # count the max times any line is repeated in full_output
            lines = full_output.split('\n')
            line_count = {}
            for line in lines:
                if line.strip() == '':
                    continue
                if line.strip() == '"""':
                    continue
                if line in line_count:
                    line_count[line] += 1
                else:
                    line_count[line] = 1
            max_count = max(line_count.values())
            rep_counts.append(max_count)

            if max_count > 20:
                repetition +=1
                # show_it = 1

                # # print the repeated line
                # for line in line_count:
                #     if line_count[line] == max_count:
                #         print ("repeated line")
                #         print (line)
                #         # fafasd

                


            

            # # if "</patch>" not in data['full_output']:
            if show_it:
                # print (data.keys())
                
                diff = extract_diff(full_output)
                minimal = extract_minimal_patch(full_output)

                len_to_show = 3000

                # if len(diff) < len(full_output):
                # if len(full_output) > 5000:

                print(f"\n\nInstance {blue(i)}")
                # print(f"Instance ID: {data['instance_id']}")
                # print(f"Prompt:\n{data['prompt']}")
                # print ('---')
                # print(f"Model output: {data['model_patch']}")
                # print(f"Model: {data['model_name_or_path']}")
                # print(f"Model path: {data['model_path']}")
                print (blue(f"Full output:"))
                print (f"{full_output[:len_to_show]}")
                print (blue(len(full_output) ))
                # print (blue(len(data['full_output']) // 3))

                print (blue(f"\nDiff:"))
                print (diff[:len_to_show])
                print (blue(len(diff)))

                # print (blue(f"\nMinimal:"))
                # print (minimal[:len_to_show])
                # print (blue(len(minimal)))
                print ('-----------------------------------------')
                print()
                # fadsfdas

                # if i > 10:
                #     break
                        
    
    print (f"Total: {total}")
    print (f"mycheck count: {mycheck}")
    print (f"rep count: {repetition}")
    # print (f"Patch count: {patch_count}")
    # print (f"Patch end count: {patch_end_count}")
    # print (rep_counts)


# output_dir = "/home/chris_cohere_ai/SWE-bench-stuff/outputs"
# run_name = 'finetuned_35B'
# output_file = os.path.join(output_dir, f"{run_name}.jsonl")
# output_file = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/finetuned_35B__SWE-bench_oracle__test.jsonl"

# output_file = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/command-r_2024_05_09__SWE-bench_oracle__test.jsonl"
output_file = "/home/chris_cohere_ai/SWE-bench-stuff/outputs/command-r-plus_2024_05_09__SWE-bench_oracle__test.jsonl"
show_some_outputs(output_file)

