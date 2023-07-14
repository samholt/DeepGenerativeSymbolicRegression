import ast
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sn
import torch
from tqdm import tqdm

pd.set_option("mode.chained_assignment", None)
SCALE = 13
HEIGHT_SCALE = 0.5
sn.set(rc={"figure.figsize": (SCALE, int(HEIGHT_SCALE * SCALE))})
sn.set(font_scale=2.0)
sn.set_style(style="white")
sn.color_palette("colorblind")

LEGEND_Y_CORD = -0.75  # * (HEIGHT_SCALE / 2.0)
SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
LEGEND_X_CORD = 0.45

plt.gcf().subplots_adjust(bottom=0.40, left=0.2, top=0.95)

PLOT_FROM_CACHE = False
PLOT_SAFTEY_MARGIN = 1.25

N = 3  # Significant Figures for Results
DP = 5

np.random.seed(999)
torch.random.manual_seed(999)


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def string_to_float_dict(d):
    return {k: float(v) if is_float(v) else v for k, v in d.items()}


def ci(data, confidence=0.95):
    # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)  # noqa: F841  # pylint: disable=unused-variable
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X_METRIC = "nevals"
Y_METRIC = "nmse_train"


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, 2 * h  # m-h, m+h


def confidence_interval(prob, n):
    return 1.96 * np.sqrt((prob * (1 - prob)) / n)


N = 5  # Significant figure

# LOG_PATH = "./process_results/results/all_run_recovery_multi-20220924-180429-fn_d_2_log_40_f.txt"
# LOG_PATH = "./process_results/results/all_run_recovery_multi-20220924-180429-l_cd_12_log_20.txt"
# LOG_PATH = "./process_results/results/all_run_recovery_multi-20220924-180429-l_cd_12_log_og_f.txt"
# LOG_PATH = "./process_results/results/all_run_recovery_multi-20220924-180429-fn_d_all_log_f.txt"
LOG_PATH = "./process_results/results/all_run_recovery_multi-20220924-180429-fn_d_5_log_existing.txt"

GENERATE_FIGS = False


if __name__ == "__main__":
    metric_to_plot = "acc_iid"
    with open(LOG_PATH) as f:
        lines = f.readlines()

    # datasets = {}
    pd_l = []
    df_tmp = []  # Drop last entry if not completed
    acc_iid = 0
    acc_ood = 0
    for line in tqdm(lines):
        if "[Test epoch=" in line:
            line = line.replace("eqs_invalid %", "eqs_invalid%")
            epoch_dict = {a.split("=")[0]: a.split("=")[1] for a in line.split("[Test")[1].strip().split() if a != "|"}
            epoch_dict["epoch"] = epoch_dict["epoch"][:-1]
            epoch_dict = string_to_float_dict(epoch_dict)
            r_best = epoch_dict["r_best"]
            epoch_dict["nmse_train"] = (1 / r_best - 1) / r_best  # pyright: ignore
            if r_best == 1.0:
                acc_iid = epoch_dict["acc_iid"]
                acc_ood = epoch_dict["acc_ood"]
            elif epoch_dict["epoch"] == 200.0:
                acc_iid = epoch_dict["acc_iid"]
                acc_ood = epoch_dict["acc_ood"]
        if "[TEST RESULT] {" in line:
            dl = [t for t in line.split("{")[1][:-2].split(", ") if "program" not in t]
            ddict = ast.literal_eval("{" + ", ".join(dl) + "}")
            dataset = ddict["dataset"]
            baseline = ddict["baseline"]
            success = ddict["success"]
            ddict["acc_iid"] = acc_iid
            ddict["acc_ood"] = acc_ood
            pd_l.append(ddict)

    dfm = pd.DataFrame(pd_l)

    if GENERATE_FIGS:
        dataset_plotting = ""
        for (dataset, baseline), group in dfm.groupby(["dataset", "baseline"]):
            if dataset_plotting == "":
                dataset_plotting = dataset
            elif dataset_plotting != dataset:
                plt.legend(
                    loc="lower center",
                    bbox_to_anchor=(LEGEND_X_CORD, LEGEND_Y_CORD),
                    ncol=1,
                    fancybox=True,
                    shadow=True,
                )
                plt.xlabel("#Evaluations")
                plt.ylabel("NMSE")
                plt.savefig(f"./results/{dataset_plotting}.png")
                plt.savefig(f"./results/{dataset_plotting}.pdf")
                plt.clf()
                dataset_plotting = dataset
            plt.plot(group[X_METRIC].to_numpy(), group[Y_METRIC].to_numpy(), label=baseline)
        plt.legend(
            loc="lower center", bbox_to_anchor=(LEGEND_X_CORD, LEGEND_Y_CORD), ncol=1, fancybox=True, shadow=True
        )
        plt.xlabel("#Evaluations")
        plt.ylabel("NMSE")
        plt.savefig(f"./results/{dataset_plotting}.png")
        plt.savefig(f"./results/{dataset_plotting}.pdf")
        plt.clf()

    n = dfm.run_seed.nunique()
    confidence_interval_data = partial(confidence_interval, n=n)

    assert not dfm.success.isnull().values.any(), "Nan values in the "  # pyright: ignore
    # .fillna(False)
    dfm["dataset"] = dfm.dataset.apply(lambda x: int(x.split("_")[-1]))

    df_out = dfm.groupby(["dataset", "baseline"]).agg([np.mean, ci]).reset_index()
    add_zero = False

    for baseline in df_out.baseline.unique():
        df_baseline = df_out[df_out.baseline == baseline]
        if add_zero:
            avg_recovery_rate = np.concatenate((df_baseline.success["mean"].to_numpy(), np.array([0]))).mean() * 100
            avg_recovery_rate_ci = (
                np.concatenate(
                    (df_baseline.success["mean"].map(confidence_interval_data).to_numpy(), np.array([0]))
                ).mean()
                * 100
            )
        else:
            avg_recovery_rate = df_baseline.success["mean"].mean() * 100
            avg_recovery_rate_ci = df_baseline.success["mean"].map(confidence_interval_data).mean() * 100
        print(f"{baseline}: {avg_recovery_rate:.2f} +/- {avg_recovery_rate_ci:.2f}")

    dfm_only_true = dfm[dfm.success == True]  # noqa: E712
    df_out = dfm.groupby(["dataset", "baseline"]).agg([np.mean, ci, np.std]).reset_index()

    average_equations_for_datasets_l = []
    for dataset in df_out.dataset.unique():
        df_dataset = df_out[df_out.dataset == dataset]
        if (df_dataset[df_dataset.baseline == "DSO"]["success"]["mean"] > 0).iloc[0] and (
            df_dataset[df_dataset.baseline == "DGSR-PRE-TRAINED"]["success"]["mean"] > 0
        ).iloc[0]:
            for baseline in df_out.baseline.unique():
                average_equations_for_datasets_l.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "n_samples_mean": df_dataset[df_dataset.baseline == baseline]["n_samples"]["mean"].iloc[0],
                        "n_samples_std": df_dataset[df_dataset.baseline == baseline]["n_samples"]["std"].iloc[0],
                        "n_samples_ci": df_dataset[df_dataset.baseline == baseline]["n_samples"]["ci"].iloc[0],
                    }
                )
    average_equations_for_datasets = pd.DataFrame(average_equations_for_datasets_l)
    for baseline in average_equations_for_datasets.baseline.unique():
        df_baseline = average_equations_for_datasets[average_equations_for_datasets.baseline == baseline]
        print(
            f"{baseline}: Avg. Eq. Evals. "
            f"{df_baseline.n_samples_mean.mean():,.2f} +/- {df_baseline.n_samples_ci.mean():,.2f}"
        )

    print("Final results now")

    a_out = dfm[["dataset", "baseline", "run_seed", "seed", "success", "n_samples"]]
    print(a_out.sort_values(by=["dataset", "baseline", "run_seed"]).reset_index().to_string())

    print(a_out.groupby(["dataset", "baseline"]).agg("mean").reset_index().to_string())
    print("")

    for index, row in df_out.iterrows():
        print(
            f"{row['dataset'].iloc[0]} {row['baseline'].iloc[0]} : {row.n_samples['mean']:,.0f} "
            # pylint: disable-next=anomalous-backslash-in-string
            f"$\pm$ {row.n_samples['ci']:,.0f}"  # noqa: W605  # pyright: ignore
        )

    print("")
