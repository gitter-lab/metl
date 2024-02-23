""" helper functions for trained model analysis"""

from os.path import join, isfile, isdir, basename

def get_eval_type(split_dir):
    eval_type = basename(split_dir).split("_")[0]
    return eval_type


def get_train_size(split_dir):
    eval_type = get_eval_type(split_dir)
    if eval_type in ["reduced", "resampled"]:
        return int(basename(split_dir).split("_")[1][2:])
    else:
        return -1


def get_split_rep_num(split_dir):
    eval_type = get_eval_type(split_dir)
    if eval_type in ["reduced", "resampled"]:
        return int(basename(split_dir).split("_")[-1])
    else:
        return -1