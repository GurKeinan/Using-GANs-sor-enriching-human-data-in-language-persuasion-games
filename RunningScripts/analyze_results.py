import pandas as pd
import wandb
from tqdm import tqdm
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.colors as mcolors

from read_wandb import wandb_results
api = wandb_results("NLP2024_PROJECT_213495260_213635899", wandb_username="west-best-dorms")

BASE_METRIC = "accuracy_per_mean_user_and_bot"


def result_metric(sweeps, group_name, drop_list=[0], drop_HPT=False, metric=BASE_METRIC, epoch="best"):
    df = api.get_sweeps_results(sweeps, metric=metric)

    config_cols = [c for c in df.columns if
                   "config_" in c and c != "config_wandb_run_id" and c != "config_online_simulation_size"]
    HPT_cols = [col for col in config_cols if df[col].nunique() > 1]
    print(HPT_cols)
    if drop_HPT:
        df = df.drop([c for c in HPT_cols if not c in ["config_LLM_SIM_SIZE", "config_seed"]], axis=1)
        HPT_cols = ["config_LLM_SIM_SIZE", "config_seed"]

    # Remove non-numeric columns before computing mean and std
    numeric_cols = df.select_dtypes(include=np.number).columns
    df_numeric = df[numeric_cols]

    grouped = df_numeric.groupby([c for c in HPT_cols if c != "config_seed"])

    mean_df = grouped.mean()
    std_df = grouped.std()

    # Re-add non-numeric columns before computing best_col
    for col in config_cols:
        if col not in mean_df.columns:
            mean_df[col] = df[col]

    if epoch == "best":
        best_col = mean_df[
            [c for c in mean_df.columns if (metric in c and metric[-4:] == c.split("_epoch")[0][-4:])]].idxmax(axis=1)
    else:
        best_col = mean_df[[c for c in mean_df.columns if f"{metric}_epoch{epoch}" in c]].idxmax(axis=1)

    result = grouped.apply(lambda x: x[best_col.loc[x.name]].values)
    means = grouped.apply(lambda x: x[best_col.loc[x.name]].mean())
    stds = grouped.apply(lambda x: x[best_col.loc[x.name]].std())

    df_cols = {'mean': means, 'std': stds, 'values': result.values}
    if epoch == "best": df_cols['epoch'] = best_col.apply(
        lambda x: int(x.split("epoch")[1]) if "epoch" in x else "last")

    df_cols['CI'] = result.apply(lambda x: bootstrap_ci(x))

    summary_df = pd.DataFrame(df_cols, index=best_col.index)
    for d in drop_list:
        if d in summary_df.index:
            summary_df = summary_df.drop(d)
    if len(summary_df.index.names) == 1:
        return summary_df.rename_axis(group_name)
    else:
        return summary_df


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means += [np.mean(sample)]
    lower_bound, upper_bound = np.percentile(bootstrapped_means,
                                                 [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return lower_bound, upper_bound


# Create the directory if it doesn't exist
directory = 'sweeps_csvs'
os.makedirs(directory) if not os.path.exists(directory) else None

# TODO: Change the sweep IDs to the ones you want to analyze
# TODO: Change the group_name to the name of the group you want to analyze
sweep_results = result_metric(['o19zqhn6', 'mqwotr8d', '5wetracq', 'd69d5s40', '6z377uko'], "GAN_simulation", drop_HPT=False, epoch="best")
print(sweep_results)
# round every number in the dataframe to 3 digits after the decimal point
sweep_results = sweep_results.round(3)
# transform to latex after dropping 'std' and 'values' columns
print(sweep_results.drop(['std', 'values'], axis=1).to_latex())