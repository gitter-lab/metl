""" functions for computing performance metrics (can add more like NDCG in the future) """

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics as skm


def compute_metrics(targets, predictions,
                    metrics=("mse", "pearsonr", "r2", "spearmanr")):
    metrics_dict = {}
    for metric in metrics:
        if np.isnan(targets).any() or np.isnan(predictions).any():
            metrics_dict[metric] = np.nan
        elif metric == "mse":
            metrics_dict["mse"] = skm.mean_squared_error(targets, predictions)
        elif metric == "pearsonr":
            metrics_dict["pearsonr"] = pearsonr(targets, predictions)[0]
        elif metric == "spearmanr":
            metrics_dict["spearmanr"] = spearmanr(targets, predictions)[0]
        elif metric == "r2":
            metrics_dict["r2"] = skm.r2_score(targets, predictions)

    return metrics_dict
