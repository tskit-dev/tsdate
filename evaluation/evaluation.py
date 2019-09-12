"""
Code to run simulations testing accuracy of tsdate

Run:
python date.py

"""

import msprime
import tsinfer
import tskit

import argparse
from itertools import combinations
import logging
import math
import multiprocessing
import os
import random
import subprocess
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(1, '../tsdate')

import tsdate # NOQA

relate_executable = os.path.join('tools', 'relate_v1.0.16_MacOSX',
                                 'bin', 'Relate')
relatefileformat_executable = os.path.join('tools', 'relate_v1.0.16_MacOSX',
                                           'bin', 'RelateFileFormats')
geva_executable = os.path.join('tools', 'geva', 'geva_v1beta')

TSDATE = "tsdate"
RELATE = "Relate"
GEVA = "GEVA"


def make_no_errors(g, error_prob):
    assert error_prob == 0
    return g


def make_seq_errors_simple(g, error_prob):
    """
    """
    raise NotImplementedError


def make_seq_errors_genotype_model(g, error_probs):
    """
    Given an empirically estimated error probability matrix, resample for a
    particular variant. Determine variant frequency and true genotype
    (g0, g1, or g2), then return observed genotype based on row in error_probs
    with nearest frequency. Treat each pair of alleles as a diploid individual.
    """
    m = g.shape[0]
    frequency = np.sum(g) / m
    closest_row = (error_probs['freq'] - frequency).abs().argsort()[:1]
    closest_freq = error_probs.iloc[closest_row]

    w = np.copy(g)

    # Make diploid (iterate each pair of alleles)
    genos = np.reshape(w, (-1, 2))

    # Record the true genotypes (0, 0=>0; 1, 0=>1; 0, 1=>2, 1, 1=>3)
    count = np.sum(np.array([1, 2]) * genos, axis=1)

    base_genotypes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    genos[count == 0, :] = base_genotypes[
        np.random.choice(
            4, sum(count == 0),
            p=closest_freq[['p00', 'p01', 'p01', 'p02']]
            .values[0] * [1, 0.5, 0.5, 1]), :]
    genos[count == 1, :] = base_genotypes[[0, 1, 3], :][
        np.random.choice(3, sum(count == 1),
                         p=closest_freq[['p10', 'p11', 'p12']].values[0]), :]
    genos[count == 2, :] = base_genotypes[[0, 2, 3], :][
        np.random.choice(3, sum(count == 2),
                         p=closest_freq[['p10', 'p11', 'p12']].values[0]), :]
    genos[count == 3, :] = base_genotypes[
        np.random.choice(4, sum(count == 3),
                         p=closest_freq[['p20', 'p21', 'p21', 'p22']]
                         .values[0]*[1, 0.5, 0.5, 1]), :]

    return np.reshape(genos, -1)


def generate_samples(
        ts, fn, aa_error="0", seq_error="0", empirical_seq_err_name=""):
    """
    Generate a samples file from a simulated ts. We can pass an integer or a
    matrix as the seq_error.
    If a matrix, specify a name for it in empirical_seq_err
    """
    record_rate = logging.getLogger().isEnabledFor(logging.INFO)
    n_variants = bits_flipped = bad_ancestors = 0
    assert ts.num_sites != 0
    fn += ".samples"
    sample_data = tsinfer.SampleData(path="tmp/" + fn,
                                     sequence_length=ts.sequence_length)

    # Setup the sequencing error used.
    # Empirical error should be a matrix not a float
    if not empirical_seq_err_name:
        seq_error = float(seq_error) if seq_error else 0
        if seq_error == 0:
            record_rate = False  # no point recording the achieved error rate
            sequencing_error = make_no_errors
        else:
            logging.info("Adding genotyping error: {} used in file {}".format(
                seq_error, fn))
            sequencing_error = make_seq_errors_simple
    else:
        logging.info("Adding empirical genotyping error: {} used in file {}"
                     .format(empirical_seq_err_name, fn))
        sequencing_error = make_seq_errors_genotype_model
    # Setup the ancestral state error used
    aa_error = float(aa_error) if aa_error else 0
    aa_error_by_site = np.zeros(ts.num_sites, dtype=np.bool)
    if aa_error > 0:
        assert aa_error <= 1
        n_bad_sites = round(aa_error*ts.num_sites)
        logging.info("""Adding ancestral allele polarity error:
                        {}% ({}/{} sites) used in file {}"""
                     .format(aa_error * 100, n_bad_sites, ts.num_sites, fn))
        # This gives *exactly* a proportion aa_error or bad sites
        # NB - to to this probabilitistically,
        # use np.binomial(1, e, ts.num_sites)
        aa_error_by_site[0:n_bad_sites] = True
        np.random.shuffle(aa_error_by_site)
        assert sum(aa_error_by_site) == n_bad_sites
    for ancestral_allele_error, v in zip(aa_error_by_site, ts.variants()):
        n_variants += 1
        genotypes = sequencing_error(v.genotypes, seq_error)
        if record_rate:
            bits_flipped += np.sum(np.logical_xor(genotypes, v.genotypes))
            bad_ancestors += ancestral_allele_error
        if ancestral_allele_error:
            sample_data.add_site(
                position=v.site.position, alleles=v.alleles,
                genotypes=1 - genotypes)
        else:
            sample_data.add_site(
                position=v.site.position, alleles=v.alleles,
                genotypes=genotypes)
    if record_rate:
        logging.info(
            " actual error rate = {} over {} sites before {} ancestors flipped"
            .format(bits_flipped/(n_variants*ts.sample_size),
                    n_variants, bad_ancestors))

    sample_data.finalise()
    return sample_data


def run_vanilla_simulation(
        sample_size, Ne, length, mutation_rate, recombination_rate, seed=None):
    """
    Run simulation
    """
    ts = msprime.simulate(
        sample_size=sample_size, Ne=Ne, length=length,
        mutation_rate=mutation_rate, recombination_rate=recombination_rate,
        random_seed=seed)
    return ts


def out_of_africa():
    # First we set out the maximum likelihood values of the various parameters
    # given in Table 1.
    N_A = 7300
    N_B = 2100
    N_AF = 12300
    N_EU0 = 1000
    N_AS0 = 510
    # Times are provided in years, so we convert into generations.
    generation_time = 25
    T_AF = 220e3 / generation_time
    T_B = 140e3 / generation_time
    T_EU_AS = 21.2e3 / generation_time
    # We need to work out the starting (diploid) population sizes based on
    # the growth rates provided for these two populations
    r_EU = 0.004
    r_AS = 0.0055
    N_EU = N_EU0 / math.exp(-r_EU * T_EU_AS)
    N_AS = N_AS0 / math.exp(-r_AS * T_EU_AS)
    # Migration rates during the various epochs.
    m_AF_B = 25e-5
    m_AF_EU = 3e-5
    m_AF_AS = 1.9e-5
    m_EU_AS = 9.6e-5
    # Population IDs correspond to their indexes in the population
    # configuration array. Therefore, we have 0=YRI, 1=CEU and 2=CHB
    # initially.
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=50, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=50, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=50, initial_size=N_AS, growth_rate=r_AS)
    ]
    migration_matrix = [
        [      0, m_AF_EU, m_AF_AS], # NOQA
        [m_AF_EU,       0, m_EU_AS],
        [m_AF_AS, m_EU_AS,       0],
    ]
    demographic_events = [
        # CEU and CHB merge into B with rate changes at T_EU_AS
        msprime.MassMigration(
            time=T_EU_AS, source=2, destination=1, proportion=1.0),
        msprime.MigrationRateChange(time=T_EU_AS, rate=0),
        msprime.MigrationRateChange(
            time=T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
        msprime.MigrationRateChange(
            time=T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
        msprime.PopulationParametersChange(
            time=T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1),
        # Population B merges into YRI at T_B
        msprime.MassMigration(
            time=T_B, source=1, destination=0, proportion=1.0),
        # Size changes to N_A at T_AF
        msprime.PopulationParametersChange(
            time=T_AF, initial_size=N_A, population_id=0)
    ]
    # Use the demography debugger to print out the demographic history
    # that we have just described.
    ts = msprime.simulate(
        population_configurations=population_configurations,
        migration_matrix=migration_matrix,
        demographic_events=demographic_events, mutation_rate=1e-8,
        recombination_rate=1e-8, length=1e5)
    return ts


def geva_age_estimate(file_name, Ne, mut_rate, rec_rate):
    """
    Perform GEVA age estimation on a given vcf
    """
    file_name = "tmp/" + file_name
    subprocess.check_output([geva_executable, "--out",
                             file_name, "--rec", str(rec_rate),
                             "--vcf", file_name + ".vcf"])
    with open(file_name+".positions.txt", "wb") as out:
        subprocess.call(["awk", "NR>3 {print last} {last = $3}",
                        file_name+".marker.txt"], stdout=out)
    try:
        subprocess.check_output(
            [geva_executable, "-i",
                file_name + ".bin", "--positions",
                file_name + ".positions.txt",
                "--hmm", "tools/geva/hmm/hmm_initial_probs.txt",
                "tools/geva/hmm/hmm_emission_probs.txt",
                "--Ne", str(Ne), "--mut", str(mut_rate),
                "--maxConcordant", "200", "--maxDiscordant",
                "200", "-o", file_name + "_estimation"])
    except subprocess.CalledProcessError as grepexc:
        print(grepexc.output)

    age_estimates = pd.read_csv(
        file_name + "_estimation.sites.txt", sep=" ", index_col="MarkerID")
    keep_ages = age_estimates[(age_estimates["Clock"] == "J")
                              & (age_estimates["Filtered"] == 1)]
    return keep_ages


def return_vcf(sample_data, filename):
    with open("tmp/"+filename+".vcf", "w") as vcf_file:
        vanilla_ts.write_vcf(vcf_file, ploidy=2)


def sampledata_to_vcf(sample_data, filename):
    """
    Input sample_data file, output VCF
    """

    num_individuals = len(sample_data.individuals_metadata[:])
    ind_list = list()
    pos_geno_dict = {"POS": list()}

    for i in range(int(num_individuals/2)):
        pos_geno_dict["msp_"+str(i)] = list()
        ind_list.append("msp_"+str(i))

    # add all the sample positions and genotypes
    for i in sample_data.genotypes():
        pos = int(round(sample_data.sites_position[i[0]]))
        if pos not in pos_geno_dict["POS"]:
            pos_geno_dict["POS"].append(pos)
            for j in range(0, len(i[1]), 2):
                pos_geno_dict["msp_" + str(int(j/2))].append(
                    str(i[1][j]) + "|" + str(i[1][j+1]))

    df = pd.DataFrame(pos_geno_dict)

    df["#CHROM"] = 1
    df["REF"] = "A"
    df["ALT"] = "T"
    df['ID'] = "."
    df['QUAL'] = "."
    df['FILTER'] = "PASS"
    df['INFO'] = "."
    df['FORMAT'] = "GT"

    cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',
            'INFO', 'FORMAT']+ind_list
    df = df[cols]

    header = """##fileformat=VCFv4.2
##source=msprime 0.6.0
##FILTER=<ID=PASS, Description="All filters passed">
##contig=<ID=1, length=""" + str(int(sample_data.sequence_length)) + """>
##FORMAT=<ID=GT, Number=1, Type=String, Description="Genotype">
"""
    output_VCF = "tmp/"+filename+".vcf"
    with open(output_VCF, 'w') as vcf:
        vcf.write(header)

    df.to_csv(output_VCF, sep="\t", mode='a', index=False)
    return df


def run_relate(ts, path_to_vcf, mut_rate, Ne, output):
    """
    Run relate software on tree sequence. Requires vcf of simulated data
    NOTE: Relate's effective population size is "of haplotypes"
    """
    # Create separate subdirectory for each run (requirement of relate)
    if not os.path.exists("tmp/" + output):
        os.mkdir("tmp/" + output)
    os.chdir("tmp/" + output)
    subprocess.check_output(["../../" + relatefileformat_executable,
                             "--mode", "ConvertFromVcf", "--haps",
                             output + ".haps",
                             "--sample", output + ".sample",
                             "-i", "../" + path_to_vcf])
    subprocess.check_output(["../../" + relate_executable, "--mode",
                             "All", "-m", str(mut_rate), "-N", str(Ne),
                             "--haps", output + ".haps",
                             "--sample", output + ".sample",
                             "--seed", "1", "-o", output, "--map",
                             "../../data/genetic_map.txt"])
    subprocess.check_output(["../../" + relatefileformat_executable, "--mode",
                             "ConvertToTreeSequence",
                             "-i", output, "-o", output])
    relate_ts = tskit.load(output + ".trees")
    table_collection = relate_ts.dump_tables()
    samples = np.repeat(1, ts.num_samples)
    internal = np.repeat(0, relate_ts.num_nodes - ts.num_samples)
    correct_sample_flags = np.array(
        np.concatenate([samples, internal]), dtype='uint32')
    table_collection.nodes.set_columns(
        flags=correct_sample_flags, time=relate_ts.tables.nodes.time)
    relate_ts_fixed = table_collection.tree_sequence()
    relate_ages = pd.read_csv(output + ".mut", sep=';')
    os.chdir("../../")
    shutil.rmtree("tmp/" + output)
    return (relate_ts_fixed, relate_ages)


def compare_mutations(method_names, ts_list, geva_ages=None, relate_ages=None):
    """
    Given a list of tree sequences, return a pandas dataframe with the age
    estimates for each mutation via each method (tsdate, tsinfer + tsdate,
    relate, geva etc.)

    :param list method_names: list of strings naming methods to be compared
    :param list ts_list: The list of tree sequences
    :param pandas.DataFrame geva_ages: mutation age estimates from geva
    :param pandas.DataFrame relate_ages: mutation age estimates from relate
    :return A DataFrame of mutations and age estimates from each method
    :rtype pandas.DataFrame
    """
    geva_included = False if geva_ages is None else True
    relate_included = False if relate_ages is None else True

    if len(method_names) != (len(ts_list) + geva_included + relate_included):
        raise ValueError("Input names of all methods to be compared")

    if not all(ts.num_mutations == ts_list[0].num_mutations for ts in ts_list):
        raise ValueError("tree sequences have unequal numbers of mutations")

    if geva_included:
        comparable_muts = geva_ages.index.values
    else:
        comparable_muts = np.arange(0, ts_list[0].num_mutations)

    def get_mut_bounds(ts):
        """
        Method to return the bounding nodes of each mutation (upper and lower)
        """
        mut_bounds = {mut: None for mut in range(ts.num_mutations)}
        for mutation in ts.mutations():
            mut_site = ts.site(mutation.site).position
            edge_num = np.intersect1d(
                np.argwhere(ts.tables.edges.child == mutation.node),
                np.argwhere(np.logical_and(ts.tables.edges.left <= mut_site,
                            ts.tables.edges.right > mut_site)))
            mut_bounds[mutation.id] = (mutation.node,
                                       ts.edge(int(edge_num)).parent)
        return mut_bounds

    mut_bounds = [get_mut_bounds(ts) for ts in ts_list]
    compare_df = pd.DataFrame(index=comparable_muts,
                              columns=method_names, dtype=float)
    for mut, row in geva_ages.iterrows():

    # for mut in comparable_muts:
        for index, ts in enumerate(ts_list):
            (child, parent) = mut_bounds[index][mut]
            child_age = ts.node(ts.mutation(mut).node).time
            parent_age = ts.node(parent).time
            true_age = np.sqrt(parent_age * child_age)
            compare_df.loc[mut, method_names[index]] = true_age 
        compare_df.loc[mut, "geva"] = row['PostMean']
        relate_row = relate_ages[relate_ages["snp"] == mut]
        compare_df.loc[mut, "relate"] = np.sqrt((relate_row['age_end'] * relate_row['age_begin']).values[0])
    #     if geva_included:
    #         compare_df.loc[mut,
    #                        method_names[len(ts_list)]] =\
    #                        geva_ages.loc[mut, 'PostMean']

    #     if relate_included:
    #         relate_row = relate_ages[relate_ages["snp"] == mut]
    #         relate_age = \
    #            np.sqrt((relate_row['age_end'] * relate_row['age_begin']).values[0])
    #         compare_df.loc[mut, method_names[len(ts_list)
    #                        + geva_included]] = relate_age 

    return compare_df


def compare_tmrcas(method_names, ts_list):
    """
    Compares pairs of TMRCA age estimates at all SNPs via different methods
    """
    sample_pairs = list(combinations(np.arange(0, ts.num_samples), 2))
    tmrcas = pd.DataFrame(index=[int(round(val))
                          for val
                          in ts.tables.sites.position],
                          columns=method_names)
    for mutation in ts.mutations():
        pos = ts.tables.sites[mutation.site].position
        for pair in sample_pairs:
            tmrcas[pos] = ts.at(pos).tmrca(pair[0], pair[1])

     


def run_tsdate(ts, n, Ne, mut_rate, time_grid):
    """
    Runs tsdate on true and inferred tree sequence
    Be sure to input HAPLOID effective population size
    """
    sample_data = tsinfer.formats.SampleData.from_tree_sequence(ts)
    inferred_ts = tsinfer.infer(sample_data)
    dated_ts = tsdate.date(ts, Ne, mutation_rate=4 * Ne * mut_rate, time_grid=time_grid)
    dated_inferred_ts = tsdate.date(inferred_ts, Ne, mutation_rate=4 * Ne * mut_rate, time_grid=time_grid)
    return dated_ts, dated_inferred_ts


def run_all_methods_compare(
        index, ts, n, Ne, mutation_rate, recombination_rate, time_grid, error_model, seed):
    """
    Function to run all comparisons and return dataframe of mutations
    """
    output = "comparison_" + str(index)
    samples = generate_samples(ts, "comparison_" + str(index))
    if error_model is not None:
        error_samples = generate_samples(ts, "error_comparison_" + str(index),
                                         empirical_seq_err_name=error_model)
    # return_vcf(samples, "comparison_" + str(index)) 
    sampledata_to_vcf(samples, "comparison_" + str(index))
    dated_ts, dated_inferred_ts = run_tsdate(ts, n, Ne/2, mutation_rate, time_grid)
    geva_ages = geva_age_estimate("comparison_" + str(index),
                                  Ne * 2, mutation_rate, recombination_rate)
    relate_output = run_relate(
        ts, "comparison_" + str(index), mutation_rate, Ne * 2, output)
    compare_df = compare_mutations(
        ["simulated_ts", "tsdate", "tsdate_inferred", "geva", "relate"],
        [ts, dated_ts, dated_inferred_ts],
        geva_ages=geva_ages, relate_ages=relate_output[1])
    return compare_df


def vanilla_tests(params):
    """
    Runs simulation and all tests for the vanilla simulation
    """
    index = int(params[0])
    n = int(params[1])
    Ne = float(params[2])
    length = int(params[3])
    mutation_rate = float(params[4])
    recombination_rate = float(params[5])
    time_grid = params[6]
    error_model = params[7]
    seed = float(params[8])
    
    ts = run_vanilla_simulation(
        n, Ne, length, mutation_rate, recombination_rate, seed)
    compare_df = run_all_methods_compare(
        index, ts, n, Ne, mutation_rate, recombination_rate, time_grid, error_model, seed)
    return compare_df


def run_multiprocessing(function, params, output, num_replicates, num_processes):
    """
    Run multiprocessing of inputted function a specified number of times
    """
    results_list = list()
    if num_processes > 1:
        logging.info("Setting up using multiprocessing ({} processes)"
                     .format(num_processes))
        with multiprocessing.Pool(processes=num_processes,
                                  maxtasksperchild=2) as pool:
            for result in pool.imap_unordered(function, params):
                #  prior_results = pd.read_csv("data/result")
                #  combined = pd.concat([prior_results, result])
                results_list.append(result)
    else:
        # When we have only one process it's easier to keep everything in the
        # same process for debugging.
        logging.info("Setting up using a single process")
        for result in map(function, params):
            results_list.append(result)
    master_df = pd.concat(results_list)
    master_df.to_csv("data/" + output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int,
                        default=10, help="number of replicates")
    parser.add_argument('num_samples', help="number of samples to simulate")
    parser.add_argument('output', help="name of output files")
    parser.add_argument('-n', '--Ne', type=float, default=10000,
                        help="effective population size")
    parser.add_argument("--length", '-l', type=int, default=1e5,
                        help="Length of the sequence")
    parser.add_argument('-m', '--mutation-rate', type=float, default=None,
                        help="mutation rate")
    parser.add_argument('-r', '--recombination-rate', type=float,
                        default=None, help="recombination rate")
    parser.add_argument('-e', '--error-model', type=str,
                        default=None, help="input error model")
    parser.add_argument('-t', '--time-grid', type=str, default="adaptive",
                        help="adaptive or uniform time grid")
    parser.add_argument(
        '--seed', '-s', type=int, default=123,
        help="use a non-default RNG seed")
    parser.add_argument(
        "--processes", '-p', type=int, default=1,
        help="number of worker processes, e.g. 40")
    args = parser.parse_args()
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    seeds = [rng.randint(1, 2**31) for i in range(args.replicates)]
    inputted_params = [int(args.num_samples), args.Ne, args.length,
                       args.mutation_rate, args.recombination_rate, args.time_grid, args.error_model]
    params = iter([np.concatenate([[index], inputted_params, [seed]])
                  for index, seed in enumerate(seeds)])
    run_multiprocessing(vanilla_tests, params, args.output, args.replicates, args.processes)


if __name__ == "__main__":
    main()
