"""Tools to evaluate generated logfiles based on log directory."""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import re
import os

import click
import pandas as pd
import seaborn as sns
import commentjson as json
from matplotlib import pyplot as plt

from dso.utils import log_and_print


class LogEval():
    """Class to hold all logged information and provide tools
    to analyze experiments."""

    PLOT_HELPER = {
        "hof": {
            "name": "Hall of Fame",
            "x_label": [
                "HoF reward distrubtion",
                "HoF error distrubtion",
                "HoF test reward distrubtion"],
            'y_label': [
                'Reward',
                'Error',
                'Test Reward'],
            "x": [
                "index",
                "index",
                "index"],
            "y": [
                "r",
                "nmse_test",
                "r_avg_test"]
        },
        "pf": {
            "name": "Pareto Front",
            "x_label": [
                "Complexity",
                "Complexity"],
            'y_label': [
                'Reward',
                'Error'],
            "x": [
                "complexity",
                "complexity"],
            "y": [
                "r",
                "nmse_test"]
        }
    }

    def __init__(self, config_path):
        """Load config, summary, hof, and pf."""
        log_and_print("-- LOADING LOGS START ----------------")
        self.warnings = []
        self.metrics = {}
        # Load config
        self.save_path = config_path
        self.config = self._get_config()
        # Load summary data (one row per seed)
        self.summary_df = self._get_summary()
        if self.summary_df is not None:
            log_and_print("Successfully loaded summary data")
        self.n_seeds = len(
            self.summary_df) if self.summary_df is not None else "N/A"
        # Load HOF
        self.hof_df = self._get_log(log_type="hof")
        if self.hof_df is not None:
            log_and_print("Successfully loaded Hall of Fame data")
        # Load pareto front
        self.pf_df = self._get_log(log_type="pf")
        if self.pf_df is not None:
            log_and_print("Successfully loaded Pareto Front data")
        # Show any warnings that occured during loading the data
        if len(self.warnings) > 0:
            log_and_print("*** WARNING:")
            for warning in self.warnings:
                log_and_print("    --> {}".format(warning))
        log_and_print("-- LOADING LOGS END ------------------")

    def _get_config(self):
        """Read the experiment's config file."""

        with open(os.path.join(self.save_path, "config.json"), 'r') as f:
            config = json.load(f)

        return config

    def _get_summary(self):
        """Read summarized benchmark data for each seed."""

        summary_df = None
        try:
            summary_path = os.path.join(self.save_path,
                                        "summary.csv")
            summary_df = pd.read_csv(summary_path)
            summary_df = summary_df.reset_index(drop=True)
            summary_df.sort_values("seed")
            try:
                self.metrics["success_rate"] = summary_df["success"].mean()
            except:
                self.metrics["success_rate"] = 0.0
        except Exception as e:
            self.warnings.append("Can't load summary: {}".format(e))

        return summary_df

    def _get_log(self, log_type):
        """Read data from log files ("hof" or "pf")."""

        log_dfs = []

        # Get files that match regexp
        task_name = self.config["experiment"]["task_name"]
        r = re.compile(r"dso_{}_\d+_{}.csv".format(task_name, log_type))
        files = filter(r.match, os.listdir(self.save_path))
        files = [os.path.join(self.save_path, f) for f in files]
        seeds = [int(f.split("_")[-2]) for f in files]

        if len(files) == 0:
            self.warnings.append("No data for {}!".format(log_type))
            return None

        # Load each df
        for f, seed in zip(files, seeds):
            df = pd.read_csv(f)
            df.insert(0, "seed", seed)
            log_dfs.append(df)

        # Combine them all
        log_df = pd.concat(log_dfs)

        # Sort HOF
        if log_type == "hof":
            log_df = log_df.sort_values(
                by=["r", "success", "seed"], ascending=False)

        # Compute PF across all runs
        if log_type == "pf":
            log_df = self._apply_pareto_filter(log_df)
            log_df = log_df.sort_values(
                by=["r", "complexity", "seed"], ascending=False)

        log_df = log_df.reset_index(drop=True)
        log_df["index"] = log_df.index

        return log_df

    def _apply_pareto_filter(self, df):
        df = df.sort_values(by=["complexity"], ascending=True)
        df = df.reset_index(drop=True)
        filtered_df = pd.DataFrame(columns=list(df))
        for index, row in df.iterrows():
            if not (filtered_df["r"] >= row["r"]).any() and \
                    not (filtered_df["complexity"] >= row["complexity"]).any() or \
                    index == 0:
                # filtered_df = pd.concat([filtered_df, row], ignore_index=True)
                filtered_df = filtered_df.append(row, ignore_index=True)
        # make sure that filtered_df has the same column types as the original df
        filtered_df = filtered_df.astype(df.dtypes.to_dict())
        return filtered_df

    def plot_results(self, results, log_type, boxplot_on=False, show_plots=False, save_plots=False):
        """Plot data from log files ("hof" or "pf")."""
        col_count = 0
        _x = []
        _y = []
        _x_label = []
        _y_label = []
        for i in range(len(self.PLOT_HELPER[log_type]["y"])):
            if self.PLOT_HELPER[log_type]["y"][i] in results:
                col_count += 1
                _x.append(self.PLOT_HELPER[log_type]["x"][i])
                _y.append(self.PLOT_HELPER[log_type]["y"][i])
                _x_label.append(self.PLOT_HELPER[log_type]["x_label"][i])
                _y_label.append(self.PLOT_HELPER[log_type]["y_label"][i])
        row_count = 2 if boxplot_on else 1
        fig, ax = plt.subplots(row_count, col_count, squeeze=0, figsize=(
            8 * col_count, 4 * row_count))
        for i in range(col_count):
            sns.lineplot(data=results, x=_x[i], y=_y[i], ax=ax[0, i])
            ax[0, i].set_xlabel(_x_label[i])
            ax[0, i].set_ylabel(_y_label[i])
            if boxplot_on:
                sns.boxplot(results[_y[i]], ax=ax[1, i])
                ax[1, i].set_xlabel(_y[i])
        plt.suptitle(
            "{} - {}".format(self.PLOT_HELPER[log_type]["name"],
                             self.config["experiment"]["task_name"]),
            fontsize=14)
        plt.tight_layout()
        if save_plots:
            save_path = os.path.join(self.save_path, "dso_{}_plot_{}.png".format(
                self.config["experiment"]["task_name"], log_type))
            log_and_print("  Saving {} plot to {}".format(
                self.PLOT_HELPER[log_type]["name"], save_path))
            plt.savefig(save_path)
        if show_plots:
            plt.show()
        plt.close()

    def analyze_log(self, show_count=5, show_hof=True, show_pf=True, show_plots=False, save_plots=False):
        """Generates a summary of important experiment outcomes."""
        log_and_print("\n-- ANALYZING LOG START --------------")
        try:
            log_and_print("Task_____________{}".format(
                self.config["task"]["task_type"]))
            log_and_print("Source path______{}".format(self.save_path))
            log_and_print("Runs_____________{}".format(self.n_seeds))
            log_and_print(
                "Max Samples/run__{}".format(self.config["training"]["n_samples"]))
            if "success_rate" in self.metrics:
                log_and_print("Success_rate_____{}".format(
                    self.metrics["success_rate"]))
            if len(self.warnings) > 0:
                log_and_print("Found issues:")
                for warning in range(len(self.warnings)):
                    log_and_print("  {}".format(warning))
            if self.hof_df is not None and show_hof:
                hof_show_count = min(show_count, len(self.hof_df))
                log_and_print('Hall of Fame (Top {} of {})____'.format(
                    hof_show_count, len(self.hof_df)))
                for i in range(hof_show_count):
                    log_and_print('  {:3d}: S={:03d} R={:8.6f} <-- {}'.format(
                        i, self.hof_df.iloc[i]['seed'], self.hof_df.iloc[i]['r'],
                        self.hof_df.iloc[i]['expression']))
                # if show_plots or save_plots:
                #     self.plot_results(
                #         self.hof_df, log_type="hof", boxplot_on=True,
                #         show_plots=show_plots, save_plots=save_plots)
            if self.pf_df is not None and show_pf:
                log_and_print('Pareto Front ({} of {})____'.format(
                    min(show_count, len(self.pf_df.index)), len(self.pf_df.index)))
                for i in range(min(show_count, len(self.pf_df.index))):
                    log_and_print('  {:3d}: S={:03d} R={:8.6f} C={:.2f} <-- {}'.format(
                        i, self.pf_df.iloc[i]['seed'], self.pf_df.iloc[i]['r'],
                        self.pf_df.iloc[i]['complexity'], self.pf_df.iloc[i]['expression']))
                # if show_plots or save_plots:
                #     self.plot_results(
                #         self.pf_df, log_type="pf",
                #         show_plots=show_plots, save_plots=save_plots)
        except ValueError:
            log_and_print("Error when analyzing!")
            for warning in self.warnings:
                log_and_print("    --> {}".format(warning))            
        except FloatingPointError:
            log_and_print("Error when analyzing!")
            for warning in self.warnings:
                log_and_print("    --> {}".format(warning))
        log_and_print("-- ANALYZING LOG END ----------------")


@click.command()
@click.argument('config_path', default=None, type=str)
@click.option('--show_count', default=10, type=int, help="Number of results we want to see from each metric.")
@click.option('--show_hof', is_flag=True, help='Show Hall of Fame results.')
@click.option('--show_pf', is_flag=True, help='Show Pareto Front results.')
@click.option('--show_plots', is_flag=True, help='Generate plots and show results as simple plots.')
@click.option('--save_plots', is_flag=True, help='Generate plots and safe to log file as simple plots.')
def main(config_path, show_count, show_hof, show_pf, show_plots, save_plots):

    log = LogEval(config_path)
    log.analyze_log(
        show_count=show_count,
        show_hof=show_hof,
        show_pf=show_pf,
        show_plots=show_plots,
        save_plots=save_plots)


if __name__ == "__main__":
    main()
