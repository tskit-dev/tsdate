import argparse
import sys
sys.path.insert(1, '../tsdate')
import tsdate
from tsdate.date import (SpansBySamples,
                         ConditionalCoalescentTimes, fill_prior, Likelihoods,
                         InOutAlgorithms, NodeGridValues, posterior_mean_var,
                         constrain_ages_topo,
                         LogLikelihoods) # NOQA
import msprime
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import tsinfer
from sklearn.metrics import mean_squared_log_error

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes



def get_prior_results():
    def evaluate_prior(ts, Ne, prior_distr, progress=False):
        fixed_node_set = set(ts.samples())
        num_samples = len(fixed_node_set)

        span_data = SpansBySamples(ts, fixed_node_set, progress=progress)
        base_priors = ConditionalCoalescentTimes(None, prior_distr)
        base_priors.add(len(fixed_node_set), False)
        mixture_prior = base_priors.get_mixture_prior_params(span_data)
        confidence_intervals = np.zeros((ts.num_nodes - ts.num_samples, 4))

        if prior_distr == 'lognorm':
            lognorm_func = scipy.stats.lognorm
            for node in np.arange(num_samples, ts.num_nodes):
                confidence_intervals[node - num_samples, 0] = np.sum(
                    span_data.get_weights(node)[num_samples].descendant_tips *
                    span_data.get_weights(node)[num_samples].weight)
                confidence_intervals[node - num_samples, 1] = 2 * Ne * lognorm_func.mean(
                    s=np.sqrt(mixture_prior[node, 1]),
                    scale=np.exp(mixture_prior[node, 0]))
                confidence_intervals[node - num_samples, 2:4] = 2 * Ne * lognorm_func.ppf(
                    [0.025, 0.975], s=np.sqrt(mixture_prior[node, 1]), scale=np.exp(
                        mixture_prior[node, 0]))
        elif prior_distr == 'gamma':
            gamma_func = scipy.stats.gamma
            for node in np.arange(ts.num_samples, ts.num_nodes):
                confidence_intervals[node - num_samples, 0] = np.sum(
                    span_data.get_weights(node)[ts.num_samples].descendant_tips *
                    span_data.get_weights(
                        node)[ts.num_samples].weight)
                confidence_intervals[node - num_samples, 1] = (2 * Ne * gamma_func.mean(
                    mixture_prior[node, 0], scale=1 / mixture_prior[node, 1]))
                confidence_intervals[node - num_samples, 2:4] = 2 * Ne * gamma_func.ppf(
                    [0.025, 0.975], mixture_prior[node, 0],
                    scale=1 / mixture_prior[node, 1])
        return(confidence_intervals)

    all_results = {i: {i: [] for i in ['in_range', 'expectations', 'real_ages',
                                       'ts_size', 'upper_bound', 'lower_bound',
                                       'num_tips']} for i in ['Lognormal_0',
                                                              'Lognormal_1e-8',
                                                              'Gamma_0', 'Gamma_1e-8']}

    for prior, (prior_distr, rec_rate) in tqdm(zip(all_results.keys(),
                                                   [('lognorm', 0), ('lognorm', 1e-8),
                                                    ('gamma', 0), ('gamma', 1e-8)]),
                                               desc="Evaluating Priors", total=4):
        for i in range(1, 11):
            Ne = 10000
            ts = msprime.simulate(sample_size=100, length=5e5, Ne=Ne, mutation_rate=1e-8,
                                  recombination_rate=rec_rate, random_seed=i)

            confidence_intervals = evaluate_prior(ts, Ne, prior_distr)
            all_results[prior]['in_range'].append(np.sum(np.logical_and(
                ts.tables.nodes.time[ts.num_samples:] < confidence_intervals[:, 3],
                ts.tables.nodes.time[ts.num_samples:] > confidence_intervals[:, 2])))
            all_results[prior]['lower_bound'].append(confidence_intervals[:, 2])
            all_results[prior]['upper_bound'].append(confidence_intervals[:, 3])
            all_results[prior]['expectations'].append(confidence_intervals[:, 1])
            all_results[prior]['num_tips'].append(confidence_intervals[:, 0])
            all_results[prior]['real_ages'].append(ts.tables.nodes.time[ts.num_samples:])
            all_results[prior]['ts_size'].append(ts.num_nodes - ts.num_samples)

    return all_results


def make_prior_plot(all_results):
    fig, ax = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = ax.ravel()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1.9, 110)
    plt.ylim(1e-3, 4e5)

    for index, ((name, result), mixtures) in enumerate(
            zip(all_results.items(), [False, False, False, False])):
        num_tips_all = np.concatenate(result['num_tips']).ravel()
        num_tips_all_int = num_tips_all.astype(int)
        only_mixtures = np.full(len(num_tips_all), True)
        if mixtures:
            only_mixtures = np.where((num_tips_all - num_tips_all_int) != 0)[0]

        upper_bound_all = np.concatenate(result['upper_bound']).ravel()[only_mixtures]
        lower_bound_all = np.concatenate(result['lower_bound']).ravel()[only_mixtures]
        expectations_all = np.concatenate(result['expectations']).ravel()[only_mixtures]

        real_ages_all = np.concatenate(result['real_ages']).ravel()[only_mixtures]
        num_tips_all = num_tips_all[only_mixtures]
        yerr = [expectations_all - lower_bound_all, upper_bound_all - expectations_all]

        axes[index].errorbar(num_tips_all, expectations_all, ls='none', yerr=yerr,
                             elinewidth=.1, alpha=0.2, color='grey',
                             label="95% credible interval of the prior")

        axes[index].scatter(num_tips_all, real_ages_all, s=1, alpha=0.5, color='blue',
                            label="True Time")
        axes[index].scatter(num_tips_all, expectations_all, s=1, color='red',
                            label="expected time", alpha=0.5)
        coverage = (np.sum(
            np.logical_and(real_ages_all < upper_bound_all,
                           real_ages_all > lower_bound_all)) / len(expectations_all))
        axes[index].text(0.35, 0.25, "Overall Coverage Probability:" +
                         "{0:.3f}".format(coverage),
                         size=10, ha='center', va='center',
                         transform=axes[index].transAxes)
        less5_tips = np.where(num_tips_all < 5)[0]
        coverage = np.sum(np.logical_and(
            real_ages_all[less5_tips] < upper_bound_all[less5_tips],
            (real_ages_all[less5_tips] > lower_bound_all[less5_tips])) / len(
            expectations_all[less5_tips]))
        axes[index].text(0.35, 0.21,
                         "<10 Tips Coverage Probability:" + "{0:.3f}".format(coverage),
                         size=10, ha='center', va='center',
                         transform=axes[index].transAxes)
        mrcas = np.where(num_tips_all == 100)[0]
        coverage = np.sum(np.logical_and(
            real_ages_all[mrcas] < upper_bound_all[mrcas],
            (real_ages_all[mrcas] > lower_bound_all[mrcas])) /
            len(expectations_all[mrcas]))
        axes[index].text(0.35, 0.17,
                         "MRCA Coverage Probability:" + "{0:.3f}".format(coverage),
                         size=10, ha='center', va='center',
                         transform=axes[index].transAxes)
        axes[index].set_title("Evaluating Conditional Coalescent Using " +
                              name.split("_")[0] + " Prior: \n 10 Samples of n=1000, \
                              length=500kb, mu=1e-8, p=" + name.split("_")[1])
        axins = zoomed_inset_axes(axes[index], 2.7, loc=7)
        axins.errorbar(num_tips_all, expectations_all, ls='none', yerr=yerr,
                       elinewidth=0.5, alpha=0.1, color='grey',
                       solid_capstyle='projecting', capsize=5,
                       label="95% credible interval of the prior")
        axins.scatter(num_tips_all, real_ages_all, s=2, color='blue', alpha=0.5,
                      label="True Time")
        axins.scatter(num_tips_all, expectations_all, s=2, color='red',
                      label="expected time", alpha=0.5)
        x1, x2, y1, y2 = 90, 105, 5e3, 3e5
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xscale('log')
        axins.set_yscale('log')
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        mark_inset(axes[index], axins, loc1=2, loc2=1, fc="none", ec="0.5")
    lgnd = axes[3].legend(loc=4, prop={'size': 12}, bbox_to_anchor=(1, -0.3))
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    lgnd.legendHandles[2]._linewidths = [2]
    fig.text(0.5, 0.04, 'Number of Tips', ha='center', size=15)
    fig.text(0.04, 0.5, 'Expectation of the Prior Distribution on Node Age',
             va='center', rotation='vertical',
             size=15)

    plt.savefig("evaluation/evaluating_conditional_coalescent_prior", dpi=300)


def evaluate_tsdate_accuracy(parameter, parameters_arr, node_mut=False, inferred=True,
                             prior_distr='lognorm', progress=True):
    Ne=10000
    if node_mut and inferred:
        raise ValueError("cannot evaluate node accuracy on inferred tree sequence")
    mutation_rate = 1e-8
    recombination_rate = 1e-8
    all_results = {i: {i: [] for i in ['io', 'max', 'true_times']} for i in list(
        map(str, parameters_arr))}

    random_seeds = range(1, 6)

    if inferred:
        inferred_progress = 'using tsinfer'
    else:
        inferred_progress = 'true topology'
    if node_mut:
        node_mut_progress = 'comparing true and estimated node times'
    else:
        node_mut_progress = 'comparing true and estimated mutation times'
    for index, param in tqdm(enumerate(parameters_arr), desc='Testing ' + parameter +
                             " " + inferred_progress + ". Evaluation by " + node_mut_progress,
                             total=len(parameters_arr),
                             disable=not progress):
        for random_seed in random_seeds:
            if parameter == 'sample_size':
                sample_size = param
            else:
                sample_size = 100
            ts = msprime.simulate(sample_size=sample_size, Ne=Ne, length=1e6,
                                  mutation_rate=mutation_rate,
                                  recombination_rate=recombination_rate,
                                  random_seed=random_seed)

            if parameter == 'length':
                ts = msprime.simulate(sample_size=sample_size, Ne=Ne, length=param,
                                      mutation_rate=mutation_rate,
                                      recombination_rate=recombination_rate,
                                      random_seed=random_seed)
            if parameter == 'mutation_rate':
                mutated_ts = msprime.mutate(ts, rate=param, random_seed=random_seed)
            else:
                mutated_ts = msprime.mutate(ts, rate=mutation_rate,
                                            random_seed=random_seed)
            if inferred:
                sample_data = tsinfer.formats.SampleData.from_tree_sequence(
                    mutated_ts, use_times=False)
                target_ts = tsinfer.infer(sample_data).simplify()
            else:
                target_ts = mutated_ts

            if parameter == 'mutation_rate':
                io_dated = tsdate.date(
                    target_ts, mutation_rate=param, Ne=Ne, progress=False,
                    method='inside_outside')
                max_dated = tsdate.date(
                    target_ts, mutation_rate=param, Ne=Ne, progress=False,
                    method='maximization')
            elif parameter == 'timepoints':
                prior = tsdate.build_prior_grid(target_ts, timepoints=param,
                                                approximate_prior=True,
                                                prior_distribution=prior_distr,
                                                progress=False)
                io_dated = tsdate.date(target_ts, mutation_rate=mutation_rate,
                                       prior=prior, Ne=Ne, progress=False,
                                       method='inside_outside')
                max_dated = tsdate.date(target_ts, mutation_rate=mutation_rate,
                                        prior=prior, Ne=Ne, progress=False,
                                        method='maximization')
            else:
                io_dated = tsdate.date(target_ts, mutation_rate=mutation_rate, Ne=Ne,
                                       progress=False, method='inside_outside')
                max_dated = tsdate.date(target_ts, mutation_rate=mutation_rate, Ne=Ne,
                                        progress=False, method='maximization')
            if node_mut and not inferred:
                all_results[str(param)]['true_times'].append(
                    mutated_ts.tables.nodes.time[ts.num_samples:])
                all_results[str(param)]['io'].append(
                    io_dated.tables.nodes.time[ts.num_samples:])
                all_results[str(param)]['max'].append(
                    max_dated.tables.nodes.time[ts.num_samples:])
            else:
                all_results[str(param)]['true_times'].append(
                    mutated_ts.tables.nodes.time[mutated_ts.tables.mutations.node])
                all_results[str(param)]['io'].append(
                    io_dated.tables.nodes.time[io_dated.tables.mutations.node])
                all_results[str(param)]['max'].append(
                    max_dated.tables.nodes.time[max_dated.tables.mutations.node])

    return all_results, prior_distr, inferred, node_mut


def plot_tsdate_accuracy(all_results, parameter, parameter_arr, prior_distr, inferred,
                         node_mut):
    f, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlim(2e-1, 2e5)
    axes[0, 0].set_ylim(2e-1, 2e5)

    for index, param in enumerate(parameter_arr):
        true_ages = np.concatenate(all_results[param]['true_times'])
        maximized = np.concatenate(all_results[param]['max'])
        inside_outside = np.concatenate(all_results[param]['io'])

        axes[index, 0].scatter(true_ages, inside_outside, alpha=0.2, s=10,
                               label="Inside-Outside")
        axes[index, 1].scatter(true_ages, maximized, alpha=0.2, s=10,
                               label="Maximized")
        axes[index, 0].plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
        axes[index, 1].plot(plt.xlim(), plt.ylim(), ls="--", c=".3")

        axes[index, 0].text(0.05, 0.9, "RMSLE: " + "{0:.2f}".format(
            mean_squared_log_error(true_ages, inside_outside)),
            transform=axes[index, 0].transAxes, size=15)
        axes[index, 1].text(0.05, 0.9, "RMSLE: " + "{0:.2f}".format(
            mean_squared_log_error(true_ages, maximized)),
            transform=axes[index, 1].transAxes, size=15)
        axes[index, 0].text(0.05, 0.8, "Pearson's r: " + "{0:.2f}".format(
            scipy.stats.pearsonr(true_ages, inside_outside)[0]),
            transform=axes[index, 0].transAxes, size=15)
        axes[index, 1].text(0.05, 0.8, "Pearson's r: " + "{0:.2f}".format(
            scipy.stats.pearsonr(true_ages, maximized)[0]),
            transform=axes[index, 1].transAxes, size=15)
        axes[index, 0].text(0.05, 0.7, "Spearman's Rho: " + "{0:.2f}".format(
            scipy.stats.spearmanr(true_ages, inside_outside)[0]),
            transform=axes[index, 0].transAxes, size=15)
        axes[index, 1].text(0.05, 0.7, "Spearman's Rho: " + "{0:.2f}".format(
            scipy.stats.spearmanr(true_ages, maximized)[0]),
            transform=axes[index, 1].transAxes, size=15)
        axes[index, 0].text(0.05, 0.6, "Bias:" + "{0:.2f}".format(
            np.mean(true_ages) - np.mean(inside_outside)),
            transform=axes[index, 0].transAxes, size=15)
        axes[index, 1].text(0.05, 0.6, "Bias:" + "{0:.2f}".format(
            np.mean(true_ages) - np.mean(maximized)),
            transform=axes[index, 1].transAxes, size=15)
        axes[index, 1].text(1.04, 0.8, parameter + ": " + str(param), rotation=90,
                            color='Red', transform=axes[index, 1].transAxes, size=20)

    axes[0, 0].set_title("Inside-Outside", size=20)
    axes[0, 1].set_title("Maximization", size=20)

    f.text(0.5, 0.05, 'True Time', ha='center', size=25)
    f.text(0.04, 0.5, 'Estimated Time', va='center',
           rotation='vertical', size=25)

    if inferred:
        inferred = "Inferred"
    else:
        inferred = "True Topologies"

    if node_mut:
        node_mut = "Nodes"
    else:
        node_mut = "Mutations"

    if parameter == 'Mut Rate':
        plt.suptitle("Evaluating " + parameter + ": " + inferred + " " + node_mut +
                     " vs. True " + node_mut + ". \n  Inside-Outside Algorithm and Maximization. \n" + prior_distr + " Prior, n=100, Length=1Mb, Rec Rate=1e-8", y=0.99, size=21)
    elif parameter == 'Sample Size':
        plt.suptitle("Evaluating " + parameter + ": " + inferred + " " + node_mut +
                     " vs. True " + node_mut + ". \n Inside-Outside Algorithm and Maximization. \n" + prior_distr + " Prior, Length=1Mb, Mut Rate=1e-8, Rec Rate=1e-8", y=0.99, size=21)
    elif parameter == 'Length':
        plt.suptitle("Evaluating " + parameter + ": " + inferred + " " + node_mut +
                     " vs. True " + node_mut + ". \n Inside-Outside Algorithm and Maximization. \n" + prior_distr + " Prior, n=100, Mut Rate=1e-8, Rec Rate=1e-8", y=0.99, size=21)
    elif parameter == 'Timepoints':
        plt.suptitle("Evaluating " + parameter + ": " + inferred + " " + node_mut +
                     " vs. True " + node_mut + ". \n Inside-Outside Algorithm and Maximization. \n" + prior_distr + " Prior, n=100, length=1Mb, Mut Rate=1e-8, Rec Rate=1e-8", y=0.99, size=21)
    # plt.tight_layout()
    plt.savefig("evaluation/" + parameter + "_" + inferred + "_" + node_mut + "_" + prior_distr +
                "_accuracy", dpi=300, bbox_inches='tight')


def run_eval(args):
    if args.prior:
        all_results = get_prior_results()
        make_prior_plot(all_results)
    if args.sample_size:
        samplesize_inf, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'sample_size', [50, 250, 500], inferred=True, progress=True)
        plot_tsdate_accuracy(samplesize_inf, "Sample Size",
                             ['50', '250', '500'], prior_distr, inferred, node_mut)
        samplesize_inf_node, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'sample_size', [50, 250, 500], inferred=False, node_mut=True, progress=True)
        plot_tsdate_accuracy(samplesize_inf_node, "Sample Size",
                             ['50', '250', '500'], prior_distr, inferred, node_mut)
        samplesize_node, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'sample_size', [50, 250, 500], inferred=False, progress=True)
        plot_tsdate_accuracy(samplesize_node, "Sample Size",
                             ['50', '250', '500'], prior_distr, inferred, node_mut)
    if args.mutation_rate:
        mutrate_inf, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'mutation_rate', [1e-09, 1e-08, 1e-07], inferred=True, progress=True)
        plot_tsdate_accuracy(mutrate_inf, "Mut Rate",
                             ['1e-09', '1e-08', '1e-07'], prior_distr, inferred,
                             node_mut)
        mutrate_inf_node, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'mutation_rate', [1e-09, 1e-08, 1e-07], inferred=False, node_mut=True,
            progress=True)
        plot_tsdate_accuracy(mutrate_inf_node, "Mut Rate",
                             ['1e-09', '1e-08', '1e-07'], prior_distr, inferred,
                             node_mut)
        mutrate_node, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'mutation_rate', [1e-09, 1e-08, 1e-07], inferred=False, progress=True)
        plot_tsdate_accuracy(mutrate_node, "Mut Rate",
                             ['1e-09', '1e-08', '1e-07'], prior_distr, inferred,
                             node_mut)

    if args.length:
        length_inf, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'length', [5e4, 5e5, 5e6], inferred=True, progress=True)
        plot_tsdate_accuracy(length_inf, "Length",
                             ['50000.0', '500000.0', '5000000.0'], prior_distr, inferred,
                             node_mut)
        length_inf_node, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'length', [5e4, 5e5, 5e6], inferred=False, node_mut=True, progress=True)
        plot_tsdate_accuracy(length_inf_node, "Length",
                             ['50000.0', '500000.0', '5000000.0'], prior_distr, inferred,
                             node_mut)
        length_node, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'length', [5e4, 5e5, 5e6], inferred=False, progress=True)
        plot_tsdate_accuracy(length_node, "Length",
                             ['50000.0', '500000.0', '5000000.0'], prior_distr, inferred,
                             node_mut)

    if args.timepoints:
        timepoints_inf, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'timepoints', [5, 10, 50], inferred=True, progress=True)
        plot_tsdate_accuracy(timepoints_inf, "Timepoints",
                             ['5', '10', '50'], prior_distr, inferred, node_mut)
        timepoints_inf_node, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'timepoints', [5, 10, 50], inferred=False, node_mut=True, progress=True)
        plot_tsdate_accuracy(timepoints_inf_node, "Timepoints",
                             ['5', '10', '50'], prior_distr, inferred, node_mut)
        timepoints_node, prior_distr, inferred, node_mut = evaluate_tsdate_accuracy(
            'timepoints', [5, 10, 50], inferred=False, progress=True)
        plot_tsdate_accuracy(timepoints_node, "Timepoints",
                             ['5', '10', '50'], prior_distr, inferred, node_mut)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tsdate.")
    parser.add_argument("--prior", action='store_true', help="Evaluate the prior")
    parser.add_argument("--sample-size", action='store_true',
                        help="Evaluate effect of variable sample size")
    parser.add_argument("--mutation-rate", action='store_true',
                        help="Evaluate effect of variable mutation rate")
    parser.add_argument("--length", action='store_true',
                        help="Evaluate effect of variable length")
    parser.add_argument("--timepoints", action='store_true',
                        help="Evaluate effect of variable numbers of timepoints")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
