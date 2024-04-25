import argparse, os, re

from multiprocessing import Pool, cpu_count
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    PatchType,
)
from swebench.harness.context_manager import TaskEnvContextManager
from swebench.harness.engine_validation import setup_testbed
from swebench.harness.utils import (
    extract_minimal_patch,
    get_instances,
    split_instances,
    DotDict
)
from tqdm.auto import tqdm



def evaluate_predictions(data: dict):
    """
    Sets up task environment context manager. Each prediction is then
    evaluated within the context manager.

    Args:
        data: Dict containing task instances and other data
            task_instances: List of [task instance, prediction] pairs to evalute
            + setup_testbed args
    """
    data_dict = DotDict(data)
    # for task_instance in tqdm(
    #     data_dict.task_instances,
    #     disable=data_dict.verbose,
    #     desc=f"Evaluating predictions for {data_dict.log_dir}"
    # ):
    # print (data_dict.log_file)
    # fasdf
    # No tqdm
    for task_instance in data_dict.task_instances:
        with TaskEnvContextManager(
            task_instance,
            data_dict.testbed,
            data_dict.venv,
            # data_dict.log_dir,
            # data_dict.log_file,
            data_dict.conda_path,
            verbose=data_dict.verbose,
            timeout=data_dict.timeout,
            is_eval=True,
            # log_suffix=data_dict.log_suffix,
        ) as tcm:
            # # Attempt to set up environment with task instance
            # if not tcm.reset_task_env(task_instance):
            #     continue

            # # Attempt to apply prediction
            # patch_type = PatchType.PATCH_PRED_TRY.value

            # # If prediction patch doesn't apply, try to do some minor patch refactoring and try again
            # if not tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type) \
            #     and task_instance[KEY_PREDICTION] is not None \
            #     and task_instance[KEY_PREDICTION] != "":
            #     task_instance[KEY_PREDICTION] = extract_minimal_patch(task_instance[KEY_PREDICTION])
            #     patch_type = PatchType.PATCH_PRED_MINIMAL_TRY.value
            #     if not tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type):
            #         # Continue if edited patch still doesn't apply
            #         continue
            # tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type, revert=True)

            # # Set prediction patch label based on whether patch was edited
            # if patch_type == PatchType.PATCH_PRED_MINIMAL_TRY.value:
            #     patch_type = PatchType.PATCH_PRED_MINIMAL.value
            # else:
            #     patch_type = PatchType.PATCH_PRED.value

            # # Run installation + testing script
            # # print (f'Count: {task_instance['count']}') 
            # if (
            #     not tcm.run_install_task(task_instance)
            #     or not tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type)
            #     or not tcm.apply_patch(task_instance["test_patch"], patch_type=PatchType.PATCH_TEST.value)
            #     or not tcm.run_tests_task(task_instance)
            # ):
            #     continue


            # My version
            if not tcm.reset_task_env(task_instance):
                continue
            if not tcm.run_install_task(task_instance):
                continue
            if not tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=PatchType.PATCH_PRED.value):
                continue
            if not tcm.run_tests_task(task_instance):
                continue





def main(args):
    """
    Splits predictions into multiple groups if num_workers > 1. Each group is
    then evaluated in parallel.
    """
    # if args.num_workers is None:
    #     args.num_workers = cpu_count()

    predictions = get_instances(args.predictions_path)

    # # Remove predictions that have already been evaluated
    # if args.skip_existing:
    #     predictions_filtered = []
    #     for p in predictions:
    #         log_file_name = f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.eval.log"
    #         if args.log_suffix is not None:
    #             log_file_name = f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.{args.log_suffix}.eval.log"
    #         path_log = os.path.join(args.log_dir, log_file_name)
    #         print (path_log)
    #         fasdfd
    #         if not os.path.exists(path_log):
    #             predictions_filtered.append(p)
    #     if len(predictions_filtered) == 0:
    #         return
    #     else:
    #         predictions = predictions_filtered

    # predictions_groups = [predictions] # split_instances(predictions, args.num_workers)

    # data_groups = [
    #     {
    #         "task_instances": g,
    #         "func": evaluate_predictions,
    #         **vars(args),
    #     }
    #     for g in predictions_groups
    # ]

    # print (f"setting up testbed for args {args.count}")

    task = {
        "task_instances": predictions,
        "func": evaluate_predictions,
        **vars(args),
    }

    # if args.num_workers == 1:
    setup_testbed(task)
    return

    # pool = Pool(processes=args.num_workers)
    # pool.map(setup_testbed, data_groups)
    # pool.close()
    # pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, help="Path to predictions instances file", required=True)
    parser.add_argument("--log_dir", type=str, help="Path to log directory", required=True)
    parser.add_argument("--conda_link", type=str, default=None, help="(Optional) URL to conda installation to use")
    parser.add_argument("--log_suffix", type=str, default=None, help="(Optional) Suffix to append to log file names")
    parser.add_argument("--num_workers", type=int, default=1, help="(Optional) Number of workers")
    parser.add_argument("--path_conda", type=str, help="(Optional) Path to miniconda3 or anaconda installation")
    parser.add_argument("--skip_existing", action="store_true", help="(Optional) Skip existing logs")
    parser.add_argument("--testbed", type=str, help="(Optional) Path to testbed directory")
    parser.add_argument("--temp_dir", type=str, help="(Optional) Path to temporary directory for storing virtual envs")
    parser.add_argument("--timeout", type=int, default=None, help="(Optional) Timeout (seconds) for testing script execution")
    parser.add_argument("--verbose", action="store_true", help="(Optional) Verbose mode")
    args = parser.parse_args()
    main(args)
