import logging
import os
from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as multiprocessing
from dso.utils import log_and_print

from config import get_config
from exp_main import top_main

conf = get_config()

conf.exp.seed_runs = 40
conf.exp.n_cores_task = 1  # 7 if GPU memory is at least 24GB, else tune to be smaller
conf.exp.seed_start = 0
conf.exp.baselines = ["DGSR-PRE-TRAINED", "NGGP", "NESYMRES", "GP"]
# User must specify the benchmark to run:
conf.exp.benchmark = "fn_d_2"  # Possible values ["fn_d_2", "fn_d_5", "l_cd_12", ""fn_d_all"]

# User must specify the pre-trained model paths
COVARS_TO_PRE_TRAINED_MODEL = {
    1: "./models/dgsr_pre_train/1_covar_koza/",
    2: "./models/dgsr_pre_train/2_covar_koza/",
    3: "./models/dgsr_pre_train/3_covar_koza/",
    4: "./models/dgsr_pre_train/4_covar_koza/",
    5: "./models/dgsr_pre_train/5_covar_koza/",
    6: "./models/dgsr_pre_train/6_covar_koza/",
    8: "./models/dgsr_pre_train/8_covar_koza/",
    12: "./models/dgsr_pre_train/12_covar_koza/",
}

PATH_TO_CHECK_IF_EXISTS = "./models/dgsr_pre_train/1_covar_koza/"

Path("./logs").mkdir(parents=True, exist_ok=True)

benchmark_df = pd.read_csv(conf.exp.benchmark_path, index_col=0, encoding="ISO-8859-1")
df = benchmark_df[benchmark_df.index.str.contains(conf.exp.benchmark)]
datasets = df.index.to_list()


file_name = os.path.basename(os.path.realpath(__file__)).split(".py")[0]
path_run_name = "all_{}-{}_01".format(file_name, conf.exp.benchmark)


def create_our_logger(path_run_name):
    logger = multiprocessing.get_logger()
    formatter = logging.Formatter("%(processName)s| %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("./logs/{}_log.txt".format(path_run_name))
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.info("STARTING NEW RUN ==========")
    logger.info(f"SEE LOG AT : ./logs/{path_run_name}_log.txt")
    return logger


logger = create_our_logger(path_run_name)
logger.info(f"See log at : ./logs/{path_run_name}_log.txt")
data_samples_to_use = int(float(df["train_spec"][0].split(",")[-1].split("]")[0]) * conf.exp.dataset_size_multiplier)


def perform_run(tuple_in):
    seed, dataset, pre_trained_model, baseline = tuple_in
    logger.info(
        f"[BASELINE_TESTING NOW] dataset={dataset} \t| baseline={baseline} \t| seed={seed} \t| data_samples={data_samples_to_use} \t| noise={conf.exp.noise}"
    )
    # try:
    if baseline == "NGGP":
        result = top_main(
            test_dataset=dataset,
            seed=seed,
            training_equations=200000,
            training_epochs=100,
            batch_outer_datasets=24,
            batch_inner_equations=100,
            pre_train=False,
            load_pre_trained_path="",
            priority_queue_training=conf.exp.priority_queue_training,
            gp_meld=conf.gp_meld.run_gp_meld,
            model="dso",
            train_path="",
            test=conf.exp.run_pool_programs_test,
            risk_seeking_pg_train=True,
            save_true_log_likelihood=conf.exp.save_true_log_likelihood,
            p_crossover=conf.gp_meld.p_crossover,
            p_mutate=conf.gp_meld.p_mutate,
            tournament_size=conf.gp_meld.tournament_size,
            generations=conf.gp_meld.generations,
            function_set=conf.exp.function_set,
            learning_rate=conf.exp.learning_rate,
            test_sample_multiplier=conf.exp.test_sample_multiplier,
            n_samples=conf.exp.n_samples,
            dataset_size_multiplier=conf.exp.dataset_size_multiplier,
            noise=conf.exp.noise,
        )
    elif baseline == "DGSR-PRE-TRAINED":
        result = top_main(
            test_dataset=dataset,
            seed=seed,
            training_equations=200000,
            training_epochs=100,
            batch_outer_datasets=24,
            batch_inner_equations=100,
            pre_train=True,
            skip_pre_training=True,
            load_pre_trained_path=pre_trained_model,
            priority_queue_training=conf.exp.priority_queue_training,
            gp_meld=conf.gp_meld.run_gp_meld,
            model="TransformerTreeEncoderController",
            train_path="",
            test=conf.exp.run_pool_programs_test,
            risk_seeking_pg_train=True,
            save_true_log_likelihood=conf.exp.save_true_log_likelihood,
            p_crossover=conf.gp_meld.p_crossover,
            p_mutate=conf.gp_meld.p_mutate,
            tournament_size=conf.gp_meld.tournament_size,
            generations=conf.gp_meld.generations,
            function_set=conf.exp.function_set,
            learning_rate=conf.exp.learning_rate,
            test_sample_multiplier=conf.exp.test_sample_multiplier,
            n_samples=conf.exp.n_samples,
            dataset_size_multiplier=conf.exp.dataset_size_multiplier,
            noise=conf.exp.noise,
        )
    elif baseline == "NESYMRES":
        result = top_main(
            test_dataset=dataset,
            seed=seed,
            training_equations=200000,
            training_epochs=100,
            batch_outer_datasets=24,
            batch_inner_equations=100,
            pre_train=False,
            skip_pre_training=True,
            load_pre_trained_path="",
            priority_queue_training=False,
            gp_meld=False,
            model="nesymres",
            train_path="",
            test=conf.exp.run_pool_programs_test,
            risk_seeking_pg_train=True,
            save_true_log_likelihood=conf.exp.save_true_log_likelihood,
            p_crossover=conf.gp_meld.p_crossover,
            p_mutate=conf.gp_meld.p_mutate,
            tournament_size=conf.gp_meld.tournament_size,
            generations=conf.gp_meld.generations,
            function_set=conf.exp.function_set,
            learning_rate=conf.exp.learning_rate,
            test_sample_multiplier=conf.exp.test_sample_multiplier,
            n_samples=conf.exp.n_samples,
            dataset_size_multiplier=conf.exp.dataset_size_multiplier,
            noise=conf.exp.noise,
        )
    elif baseline == "GP":
        result = top_main(
            test_dataset=dataset,
            seed=seed,
            training_equations=200000,
            training_epochs=100,
            batch_outer_datasets=24,
            batch_inner_equations=100,
            pre_train=False,
            skip_pre_training=True,
            load_pre_trained_path="",
            priority_queue_training=False,
            gp_meld=False,
            model="gp",
            train_path="",
            test=True,
            risk_seeking_pg_train=True,
            save_true_log_likelihood=conf.exp.save_true_log_likelihood,
            p_crossover=conf.gp_meld.p_crossover,
            p_mutate=conf.gp_meld.p_mutate,
            tournament_size=conf.gp_meld.tournament_size,
            generations=conf.gp_meld.generations,
            function_set=conf.exp.function_set,
            learning_rate=conf.exp.learning_rate,
            test_sample_multiplier=conf.exp.test_sample_multiplier,
            n_samples=conf.exp.n_samples,
            dataset_size_multiplier=conf.exp.dataset_size_multiplier,
            noise=conf.exp.noise,
        )
    result["baseline"] = baseline  # pyright: ignore
    result["run_seed"] = seed  # pyright: ignore
    result["dataset"] = dataset  # pyright: ignore
    log_and_print(f"[TEST RESULT] {result}")  # pyright: ignore
    return result  # pyright: ignore

    # except FileNotFoundError as e:
    #     logger.exception(f'[Error] {e}')
    #     log_and_print(f"[FAILED BASELINE_TESTING] dataset={dataset} \t|
    # baseline={baseline} \t| seed={seed} \t | error={e}")
    #     traceback.print_exc()
    #     raise e
    # except Exception as e:
    #     logger.exception(f'[Error] {e}')
    #     log_and_print(f"[FAILED BASELINE_TESTING] dataset={dataset} \t|
    # baseline={baseline} \t| seed={seed} \t | error={e}")
    #     traceback.print_exc()


def main(dataset, n_cores_task=conf.exp.n_cores_task):
    if not os.path.exists(PATH_TO_CHECK_IF_EXISTS):
        print("Path does not exist.")
        raise ValueError("Path does not exist.")
    task_inputs = []
    for seed in range(conf.exp.seed_start, conf.exp.seed_start + conf.exp.seed_runs):
        for baseline in conf.exp.baselines:
            task_inputs.append((seed, dataset, pre_trained_model, baseline))

    if n_cores_task is None:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task >= 2:
        pool_outer = multiprocessing.Pool(n_cores_task)
        for i, result in enumerate(pool_outer.imap(perform_run, task_inputs)):
            log_and_print(
                "INFO: Completed run {} of {} in {:.0f} s | LATEST TEST_RESULT {}".format(
                    i + 1, len(task_inputs), result["t"], result
                )
            )
    else:
        for i, task_input in enumerate(task_inputs):
            result = perform_run(task_input)
            log_and_print(
                "INFO: Completed run {} of {} in {:.0f} s | LATEST TEST_RESULT {}".format(
                    i + 1, len(task_inputs), result["t"], result
                )
            )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    from dso.config import load_config
    from dso.task import set_task

    from config import (
        dsoconfig_factory,
        nesymres_dataset_config_factory,
        nesymres_function_set_factory,
        nesymres_train_config_factory,
    )

    dsoconfig = dsoconfig_factory()
    log_and_print(df.to_string())
    for dataset, row in df.iterrows():
        covars = row["variables"]
        try:
            pre_trained_model = COVARS_TO_PRE_TRAINED_MODEL[covars]
        except KeyError:
            # pylint: disable-next=raise-missing-from
            raise ValueError(
                f"No pre-trained model in folder './models/pre_train/' for covars={covars}. "
            )
            # pre_trained_model = ""
        nesymres_dataset_config = nesymres_dataset_config_factory()
        nesymres_train_config = nesymres_train_config_factory()
        nesymres_function_set = nesymres_function_set_factory()
        dsoconfig["task"]["dataset"] = dataset
        config = load_config(dsoconfig)
        set_task(config["task"])
        try:
            main(dataset)
        except FileNotFoundError as e:
            # pylint: disable-next=raise-missing-from
            if 'nesymres_pre_train' in str(e):
                raise FileNotFoundError(
                    f"Please download the baseline pre-trained models for NeuralSymbolicRegressionThatScales from https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales and put them into the folder `models/nesymres_pre_train`. No pre-trained model of {e.filename} in folder './models/pre_train/' for covars={covars}. "
                )
            else:                
                raise FileNotFoundError(
                    f"No pre-trained model of {e.filename} in folder './models/pre_train/' for covars={covars}. "
                )
    logger.info("Fin.")
