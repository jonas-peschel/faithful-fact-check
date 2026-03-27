import argparse 
import warnings
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import List, Tuple
from numpy.typing import NDArray
from utils import load_json, order_results, sort_legend, METH2COL, METH2LABEL

def parse_args():

    parser = argparse.ArgumentParser(description="Plot results for ContextCite top-k log-prob drop metric.")
    parser.add_argument("results_path", type=str, help="Path to the file where attribution scores and experiment results (metrics) are stored.")
    parser.add_argument("plots_savepath", type=str, help="Path where to save the generated plots.")
    parser.add_argument("--use_longcite", action="store_true", help="Whether to plot top-k log-prob drop with k=#citations from LongCite model.")
    parser.add_argument("--excluded_attr_methods", type=str, nargs="*", choices=["context_cite_32", "context_cite_64", "context_cite_128", "context_cite_256", "semantic_similarity", "leave_one_out", "nli_post_hoc_naive", "nli_post_hoc_sliding_window_3", "nli_post_hoc_sliding_window_5", "nli_post_hoc_greedy_sampling", "llm_post_hoc", "longcite_llm_direct"], default=[], help="Attribution methods not to include in the plot.")
    parser.add_argument("--plot_title", type=str, default="Top-k Log-Probability Drop Metric", help="Title for the plot.")
    parser.add_argument("--ks", type=int, nargs="+", choices=range(1,10), default=None, help="For which k's to plot the results.")
    parser.add_argument("--is_error_bars", action="store_true", help="Whether to plot the error bars with the standard error of the mean.")
    parser.add_argument("--is_combine", action="store_true", help="Whether to plot combined plot with original ks and k_longcite")

    return parser.parse_args()

def aggregate_log_prob_drops(results: dict, ks: List[int] | None, attr_methods: List[str]):
    """
    Calculate mean log-prob drop over all sentences in the data for
    each answer attribution method and number of ablations k.

    Args:
        results (Dict):
            Contains the results from metrics computations.
        ks (List[int] | None):
            Numbers of context ablations for which the metric has been computed. 
            None if we use LongCite with only one k with k=#citations by LongCite model.
        attr_methods (List[str]):
            Names of attribution methods for which to calculate top-k log-prob drop.

    Returns:
        mean_drops (NDArray[float]):
            Mean log-prob drop values. Shape: (n_methods, n_ks)
        sem_drops (NDArray[float]):
            Standard error of the mean for the log-prob drop values. Shape: (n_methods, n_ks)
        mask (NDArray[bool]):
            Which sentences were dropped due to invalid data.
            Shape: (n_sentences,)
    """

    drops = []  # (n_methods, n_Ks, n_sentences)
    for method in attr_methods:
        drops_method = []   # (n_Ks, n_sentences)
        for data_point_result in results["results"]:
            if "metrics" in data_point_result["methods"][method]:
                top_k_drop_dict = data_point_result["methods"][method]["metrics"]["top_k_drop"]
            else:  # skip missing data
                continue
            top_k_drops = []
            if not ks:
                top_k_drops.append(top_k_drop_dict["top_k_drop_longcite"])
            else:
                for k in ks:
                    top_k_drops.append(top_k_drop_dict[f"top_{k}_drop"])    
            top_k_drops = np.array(top_k_drops, ndmin=2) # top_k_drops: (n_Ks, n_sentences in this data point)
            drops_method.append(top_k_drops)

        # concatenate for all data_points for the method in drops_method
        drops_method = np.concat(drops_method, axis=1)  # (n_Ks, n_sentences)
        drops.append(drops_method)

    drops = np.array(drops, dtype=float) # (n_methods, n_Ks, n_sentences); # using dtype=float to convert None values back to np.nan
    # delete any columns (i.e. answer sentences) for which at least one entry is invalid (NaN)
    mask = np.all(~np.isnan(drops), axis=(0,1))
    print(f"Dropped {drops.shape[2]-(mask.sum())}/{drops.shape[2]} sentences with NaN values.")
    drops = drops[:,:,mask]

    mean_drops = drops.mean(axis=2) # (n_methods, n_Ks)
    sem_drops = drops.std(axis=2) / np.sqrt(drops.shape[2]) # sem: standard error of the mean

    return mean_drops, sem_drops, mask

def count_longcite_citations(results: dict, mask: NDArray[np.bool_]):

    k_citations = []
    for data_point_result in results["results"]: 
        for citations in data_point_result["methods"]["longcite_llm_direct"]["citations"]:
            k_citations.append(len(citations))
    k_citations = np.array(k_citations)
    k_citations = k_citations[mask]
    mean, std = np.mean(k_citations), np.std(k_citations)
    return (mean, std)

#--- Plotting Functions ---#
def plot_top_k_log_prob_drop(mean_drops: NDArray[np.floating], sem_drops: NDArray[np.floating], labels: List[str], 
                             ks: List[int] | None, k_longcite_mean_std: Tuple[float,float] | None=None, is_error_bars: bool=False, 
                             title: str="Top-k Log-Probability Drop Metric"):
    """
    Plot bar plot of top-k log-prob drop metric.

    Args:
        mean_drops (NDArray[np.floating]): 
            Top-k log-probability drop for different attribution methods and different k.
            Mean values over data points and answer sentences.
            Shape: (n_methods, n_ks)
        sem_drops (NDArray[np.floating]):
            Standard error of the mean for the top-k log-probability drop for different
            attribution methods and different k.
            Shape: (n_methods, n_ks)
        labels (List[str]):
            Names of the different attribution methods.
        ks (List[int] | None):
            Numbers of context ablations for which the metric has been computed.
            None if we use LongCite with only one k with k=#citations by LongCite model.
        k_longcite_mean_std (Tuple[float,float] | None, optional):
            Mean and standard deviation of how many sources were cited by LongCite model
            for all sentences/statements.
        is_error_bars (bool, optional):
            Whether to plot error bars with the standard error of the mean.
            Defaults to False. 
        title (str, optional):
            Title of the plot.
    """
    
    fig, ax = plt.subplots()

    x = np.arange(mean_drops.shape[1])
    bar_width = 1 / (mean_drops.shape[0] + 1)

    mean_drops, sem_drops, labels = order_results(mean_drops, sem_drops, labels)  # ordering for the plot

    multiplier = 0
    for mean, sem, label in zip(mean_drops, sem_drops, labels):
        offset = bar_width * multiplier 
        if is_error_bars:
            rects = ax.bar(x+offset, mean, bar_width, label=METH2LABEL[label], 
                        edgecolor="white", linewidth=0.5, color=METH2COL[label], yerr=sem, capsize=2, error_kw={"ecolor": "black", "lw": 1.0})
        else:
            rects = ax.bar(x+offset, mean, bar_width, label=label, edgecolor="white", linewidth=0.5, color=METH2COL[label])
        multiplier += 1

    ax.set_title(title)
    ax.set_ylabel("Log-prob drop")
    if ks:
        xticks_labels = [f"k={k}" for k in ks]
    else:
        k_citations_mean, k_citations_std = k_longcite_mean_std
        xticks_labels = [fr"$k_{{\mathrm{{LongCite}}}} (\approx {k_citations_mean:.1f} \pm {k_citations_std:.1f})$"]
    ax.set_xticks(x + bar_width*(mean_drops.shape[0]-1)/2, xticks_labels)
    ax.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)  # place legend outside of plot
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return fig

def plot_top_k_log_prob_drop_combined(mean_drops_list, sem_drops_list, labels_list, ks, k_longcite_mean_std, is_error_bars, title):

    fig, ax = plt.subplots()

    bar_width = 1.0
    gap_size = 1.0
    curr_x = 0.0
    tick_positions = []
    seen_labels = set() 

    for group_means, group_sems, group_labels in zip(mean_drops_list, sem_drops_list, labels_list):
        group_start = curr_x
        group_means, group_sems, group_labels = order_results(group_means, group_sems, group_labels)

        for i, (mean, sem, label) in enumerate(zip(group_means, group_sems, group_labels)):
            x_pos = group_start + (i+0.5)*bar_width
            plot_label = METH2LABEL[label] if label not in seen_labels else None
            seen_labels.add(label)

            ax.bar(x_pos, mean, bar_width, label=plot_label, color=METH2COL[label], edgecolor="white", 
                linewidth=0.5, yerr=sem if is_error_bars else None, capsize=2, error_kw={"ecolor": "black", "lw": 0.5})
            
            curr_x += bar_width 

        group_end = curr_x - bar_width 
        tick_positions.append((group_start+group_end)/2)

        # add gap 
        curr_x += gap_size * bar_width 
            
    ax.set_title(title)
    ax.set_ylabel("Log-prob drop")

    k_longcite_mean, k_longcite_std = k_longcite_mean_std
    xticks_labels = [f"$k={k}$" if k != k_longcite_mean else fr"$k \approx {k_longcite_mean:.1f} \pm {k_longcite_std:.1f}$" for k in ks]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(xticks_labels)

    # sort legend
    sorted_handles, sorted_labels = sort_legend(ax)
    ax.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)  # place legend outside of plot

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return fig

#--- Plotting Functions End ---#

def main(config=None):

    if config is None:
        config = parse_args()

    # load metrics results
    results_path = Path(config.results_path)
    results = load_json(results_path)

    if config.use_longcite:
        config.ks = None 
    else:  # use all ks from the data if not provided
        if config.ks is None:
            # good luck trying to read this
            config.ks = [int(match.group(1)) for match in [re.compile(r"top_(\d+)_drop").match(key) for key in list(list(results["results"][0]["methods"].values())[0]["metrics"]["top_k_drop"].keys())] if match]

    attr_methods = list(results["results"][0]["methods"].keys())
    for excluded_attr_method in config.excluded_attr_methods:
        if excluded_attr_method in attr_methods:
            attr_methods.remove(excluded_attr_method)
        else:
            warnings.warn(f"Tried to exclude method {excluded_attr_method} but it was never included.")

    if not config.is_combine:
        # aggregate mean and standard error of the mean over the data
        mean_drops, sem_drops, mask = aggregate_log_prob_drops(results, config.ks, attr_methods)

        k_longcite_mean_std = count_longcite_citations(results, mask) if config.use_longcite else None

        # plot
        fig = plot_top_k_log_prob_drop(mean_drops, sem_drops, labels=attr_methods, ks=config.ks, 
                                    k_longcite_mean_std=k_longcite_mean_std, is_error_bars=config.is_error_bars, title=config.plot_title)
        
    elif config.is_combine:
        attr_methods_wo_longcite = attr_methods.copy() 
        attr_methods_wo_longcite.remove("longcite_llm_direct")

        ## aggregate mean and standard error of the mean over the data
        # 1. for k = 1,3,5 (use all attr_methods including LongCite to drop the same sentences with NaN values)
        mean_drops, sem_drops, _ = aggregate_log_prob_drops(results, config.ks, attr_methods)
        # remove longcite results for k=1,3,5 as they are invalid
        idx_longcite = attr_methods.index("longcite_llm_direct")
        mask = np.ones(len(attr_methods), dtype=bool)
        mask[idx_longcite] = False 
        mean_drops, sem_drops = mean_drops[mask], sem_drops[mask]

        # 2. for k = k_longcite 
        mean_drops_k_longcite, sem_drops_k_longcite, mask_k_longcite = aggregate_log_prob_drops(results, None, attr_methods)

        k_longcite_mean_std = count_longcite_citations(results, mask_k_longcite)
        k_longcite_mean, _ = k_longcite_mean_std

        # make list of mean_drops, sem_drops, and labels
        ks_combined = sorted(config.ks+[k_longcite_mean.item()])
        mean_drops_list, sem_drops_list, labels_list = [], [], []
        i = 0
        for k in ks_combined:
            if k == k_longcite_mean:
                mean_drops_list.append(mean_drops_k_longcite.squeeze())
                sem_drops_list.append(sem_drops_k_longcite.squeeze())
                labels_list.append(attr_methods)
            else:
                mean_drops_list.append(mean_drops[:,i])
                sem_drops_list.append(sem_drops[:,i])
                labels_list.append(attr_methods_wo_longcite)
                i += 1

        # plot
        fig = plot_top_k_log_prob_drop_combined(mean_drops_list, sem_drops_list, labels_list=labels_list, ks=ks_combined, 
                                                k_longcite_mean_std=k_longcite_mean_std, is_error_bars=config.is_error_bars, title=config.plot_title)

    # save results
    plots_savepath = Path(config.plots_savepath)
    plots_savepath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(plots_savepath, bbox_inches="tight")
    print(f"Saved results to: {plots_savepath}")

if __name__ == "__main__":
    main()