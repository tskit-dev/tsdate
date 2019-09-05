"""
Code to run simulations testing accuracy of tsdate

Run:
python test_accuracy.py

"""

import msprime
import tsinfer
import tskit

import argparse
import logging
import multiprocessing
import os
from tqdm import tqdm
import subprocess
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import sys

import numpy as np
import pandas as pd

sys.path.insert(1, '/Users/anthonywohns/Documents/mcvean_group/age_inference/tsdate')

import tsdate

relate_executable = os.path.join('tools', 'relate_v1.0.16_MacOSX', 'Relate')
relatefileformat_executable = os.path.join('tools', 'bin', 'RelateFileFormats')
geva_executable = os.path.join('tools', 'geva', 'geva_v1beta')

TSDATE = "tsdate"
RELATE = "Relate"
GEVA = "GEVA"

path = "/Users/anthonywohns/Documents/mcvean_group/age_inference/tsdate/"

def make_seq_errors_genotype_model(g, error_probs):
    """
    Given an empirically estimated error probability matrix, resample for a particular
    variant. Determine variant frequency and true genotype (g0, g1, or g2),
    then return observed genotype based on row in error_probs with nearest
    frequency. Treat each pair of alleles as a diploid individual.
    """
    m = g.shape[0]
    frequency = np.sum(g) / m
    closest_row = (error_probs['freq']-frequency).abs().argsort()[:1]
    closest_freq = error_probs.iloc[closest_row]

    w = np.copy(g)
    
    # Make diploid (iterate each pair of alleles)
    genos = np.reshape(w,(-1,2))

    # Record the true genotypes (0,0=>0; 1,0=>1; 0,1=>2, 1,1=>3)
    count = np.sum(np.array([1,2]) * genos,axis=1)
    
    base_genotypes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    
    genos[count==0,:]=base_genotypes[
        np.random.choice(4,sum(count==0), p=closest_freq[['p00', 'p01','p01', 'p02']].values[0]*[1,0.5,0.5,1]),:]
    genos[count==1,:]=base_genotypes[[0,1,3],:][
        np.random.choice(3,sum(count==1), p=closest_freq[['p10', 'p11', 'p12']].values[0]),:]
    genos[count==2,:]=base_genotypes[[0,2,3],:][
        np.random.choice(3,sum(count==2), p=closest_freq[['p10', 'p11', 'p12']].values[0]),:]
    genos[count==3,:]=base_genotypes[
        np.random.choice(4,sum(count==3), p=closest_freq[['p20', 'p21', 'p21', 'p22']].values[0]*[1,0.5,0.5,1]),:]

    return(np.reshape(genos,-1))

def generate_samples(
    ts, fn, aa_error="0", seq_error="0", empirical_seq_err_name=""):
    """
    Generate a samples file from a simulated ts. We can pass an integer or a 
    matrix as the seq_error. If a matrix, specify a name for it in empirical_seq_err
    """
    record_rate = logging.getLogger().isEnabledFor(logging.INFO)
    n_variants = bits_flipped = bad_ancestors = 0
    assert ts.num_sites != 0
    fn += ".samples"
    sample_data = tsinfer.SampleData(path=fn, sequence_length=ts.sequence_length)
    
    # Setup the sequencing error used. Empirical error should be a matrix not a float
    if not empirical_seq_err_name:
        seq_error = float(seq_error) if seq_error else 0
        if seq_error == 0:
            record_rate = False # no point recording the achieved error rate
            sequencing_error = make_no_errors
        else:
            logging.info("Adding genotyping error: {} used in file {}".format(
                seq_error, fn))
            sequencing_error = make_seq_errors_simple
    else:
        logging.info("Adding empirical genotyping error: {} used in file {}".format(
            empirical_seq_err_name, fn))
        sequencing_error = make_seq_errors_genotype_model
    # Setup the ancestral state error used
    aa_error = float(aa_error) if aa_error else 0
    aa_error_by_site = np.zeros(ts.num_sites, dtype=np.bool)
    if aa_error > 0:
        assert aa_error <= 1
        n_bad_sites = round(aa_error*ts.num_sites)
        logging.info("Adding ancestral allele polarity error: {}% ({}/{} sites) used in file {}"
            .format(aa_error * 100, n_bad_sites, ts.num_sites, fn))
        # This gives *exactly* a proportion aa_error or bad sites
        # NB - to to this probabilitistically, use np.binomial(1, e, ts.num_sites)
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
                position=v.site.position, alleles=v.alleles, genotypes=1-genotypes)
        else:
            sample_data.add_site(
                position=v.site.position, alleles=v.alleles, genotypes=genotypes)
    if record_rate:
        logging.info(" actual error rate = {} over {} sites before {} ancestors flipped"
            .format(bits_flipped/(n_variants*ts.sample_size), n_variants, bad_ancestors))

    sample_data.finalise()
    return sample_data

def geva_age_estimate(file_name, Ne, mut_rate, rec_rate):
    """
    Perform GEVA age estimation on a given vcf
    """
    path_geva = "/Users/anthonywohns/Documents/mcvean_group/age_inference/importance_sampling/" 
    subprocess.check_output([path_geva + "geva/geva_v1beta", "--out", path + "tmp/" + file_name, "--rec", str(rec_rate), "--vcf", path + "tmp/" + file_name + ".vcf"])
    with open(path + "tmp/"+file_name+".positions.txt","wb") as out:
        subprocess.call(["awk", "NR>3 {print last} {last = $3}", path + "tmp/"+file_name+".marker.txt"], stdout=out)
    try:
        subprocess.check_output([path_geva + "geva/./geva_v1beta", "-i", path + "tmp/"+ file_name+".bin", "--positions", path + "tmp/"+file_name+".positions.txt","--hmm", path_geva + "geva/hmm/hmm_initial_probs.txt", path_geva + "geva/hmm/hmm_emission_probs.txt","--Ne", str(Ne), "--mut", str(mut_rate), "--maxConcordant","200","--maxDiscordant", "200","-o", path + "tmp/"+file_name+"_estimation"])
    except subprocess.CalledProcessError as grepexc:
        print(grepexc.output)
        
    age_estimates = pd.read_csv(path + "tmp/"+file_name+"_estimation.sites.txt", sep = " ")
    return(age_estimates)

def samplesdata_to_ages(sample_data, Ne, length, mut_rate, rec_rate, filename):
    """
    Input sample_data file, output GEVA age estimats
    """

    num_individuals = len(sample_data.individuals_metadata[:])
    ind_list = list()
    pos_geno_dict = {"POS":list()}

    for i in range(int(num_individuals/2)):
        pos_geno_dict["msp_"+str(i)] = list()
        ind_list.append("msp_"+str(i))

    #add all the sample positions and genotypes
    for i in sample_data.genotypes():
        pos = int(round(sample_data.sites_position[i[0]]))
        if pos not in pos_geno_dict["POS"]:
            pos_geno_dict["POS"].append(pos)
            for j in range(0,len(i[1]),2):
                pos_geno_dict["msp_"+str(int(j/2))].append(str(i[1][j]) + "|" + str(i[1][j+1]))

    df = pd.DataFrame(pos_geno_dict)

    df["#CHROM"] = 1
    df["REF"] = "A"
    df["ALT"] = "T"
    df['ID'] = "."
    df['QUAL'] = "."
    df['FILTER'] = "PASS"
    df['INFO'] = "."
    df['FORMAT'] = "GT"

    cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO','FORMAT']+ind_list
    df = df[cols] 

    header = """##fileformat=VCFv4.2
##source=msprime 0.6.0
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=1,length=""" + str(int(length)) +  """>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
"""
    
    output_VCF = path + "tmp/"+filename+".vcf"
    with open(output_VCF, 'w') as vcf:
        vcf.write(header)

    df.to_csv(output_VCF, sep="\t", mode='a',index=False)

    return(geva_age_estimate(filename, Ne, mut_rate, rec_rate))


import math
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
        [      0, m_AF_EU, m_AF_AS],
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
    return(ts)


def run_simulation(ts, n, Ne, theta, rho):

    sample_data = tsinfer.formats.SampleData.from_tree_sequence(ts)
    inferred_ts = tsinfer.infer(sample_data) 
    dated_ts = tsdate.age_inference(ts,theta=theta, rho=rho)
    dated_inferred_ts_mut = tsdate.age_inference(inferred_ts, theta=theta, rho=rho)
    return(ts, dated_ts, dated_inferred_ts_mut)
    
def compare_mutations(ts, dated_ts, dated_inferred_ts_mut, keep_ages, relate_age):
    def get_mutation_child_parent(ts):
        mutation_child_parent = {mutation: None for mutation in range(ts.num_mutations)}
        for mutation in ts.mutations():
            focal_site = ts.site(mutation.site).position
            edge_num = np.intersect1d(np.argwhere(ts.tables.edges.child == mutation.node),
                           np.argwhere(np.logical_and(ts.tables.edges.left <= focal_site,
                                                      ts.tables.edges.right > focal_site)))
            mutation_child_parent[mutation.id] = (mutation.node, ts.edge(int(edge_num)).parent)
        return(mutation_child_parent)

    mutation_child_parent_ts = get_mutation_child_parent(ts)
    mutation_mapping_dated = get_mutation_child_parent(dated_ts) 
    mutation_mapping_inferred = get_mutation_child_parent(dated_inferred_ts_mut) 
    
    
    truth = list()
    dated = list()
    inferred_dated = list()
    geva = list()
    relate = list()
    for index, row in keep_ages.iterrows():
        
        (child, parent) = mutation_child_parent_ts[index]
        child_age = ts.node(ts.mutation(index).node).time
        parent_age = ts.node(parent).time
        true_age = (parent_age + child_age)/2
        truth.append(true_age)
        relate_record = relate_age[relate_age["snp"] == index]
        relate.append((relate_record['age_end'] - relate_record['age_begin']).values[0]/2)

        (child_dated, parent_dated) = mutation_mapping_dated[index]
        child_dated_age = dated_ts.node(dated_ts.mutation(index).node).time
        parent_dated_age = dated_ts.node(parent_dated).time
        tsdate_age = (parent_dated_age + child_dated_age)/2
        dated.append(tsdate_age)
        
        (child_inferred_dated, parent_inferred_dated) = mutation_mapping_inferred[index]
        child_dated_age = dated_inferred_ts_mut.node(dated_inferred_ts_mut.mutation(index).node).time
        parent_dated_age = dated_inferred_ts_mut.node(parent_inferred_dated).time
        inferred_tsdate_age = (parent_dated_age + child_dated_age)/2
        inferred_dated.append(inferred_tsdate_age)
        geva.append(row['PostMean']) 

    compare_dict = {'truth': truth,'tsdate': dated, 'tsdate_inferred': inferred_dated, 'relate': relate, 'geva': geva}
    compare_dict = pd.DataFrame(compare_dict)
    return(compare_dict)


def test_accuracy(reps):
    n = 100 
    Ne = 10000
    mut_rate = 1e-8
    rec_rate = 1e-8
    theta = 4*10000*mut_rate
    rho = 4*10000*rec_rate
    length = 1e5 
    
    compare_df_master = pd.DataFrame(columns = ['truth', 'tsdate', 'tsdate_inferred', 'relate', 'geva'])

    for rep in range(reps): 
        vanilla_ts = msprime.simulate(sample_size=n, Ne=Ne, mutation_rate=mut_rate, recombination_rate=rec_rate, length=length)

        ts, dated_ts, dated_inferred_ts_mut = run_simulation(vanilla_ts, n, Ne, theta, rho)

        # Run GEVA
        samples = generate_samples(ts, 'testing')
        ages = samplesdata_to_ages(samples, Ne=Ne, length=length, mut_rate=mut_rate, rec_rate=rec_rate, filename=str("test"))

        # Run Relate on simulated data
        relate_path = "/Users/anthonywohns/Documents/mcvean_group/software/relate_v1.0.13_MacOSX/"

        def run_relate(ts, relate_path):
            subprocess.check_output([relate_path + "bin/RelateFileFormats", "--mode", "ConvertFromVcf", "--haps", relate_path + "age_compare/compare.haps", "--sample", relate_path + "age_compare/compare.sample", "-i", path + "tmp/test"])
            subprocess.check_output([relate_path + "bin/Relate", "--mode", "All", "-m", str(mut_rate), "-N", "20000", "--haps", relate_path + "age_compare/compare.haps", "--sample", relate_path + "age_compare/compare.sample", "--seed", "1", "-o", "compare", "--map", relate_path + "genetic_map.txt"])
            subprocess.check_output(
                [relate_path + "bin/RelateFileFormats", "--mode",
                    "ConvertToTreeSequence",
                    "-i", "compare", "-o", "compare"])
        run_relate(ts, relate_path)
        relate_ts = tskit.load('compare.trees')
        table_collection = relate_ts.dump_tables()
        table_collection.nodes.flags[0:n] = 1
        table_collection = relate_ts.dump_tables()
        table_collection.nodes.set_columns(
            flags=np.array(np.concatenate(
                [np.repeat(1, n),
                 np.repeat(0, relate_ts.num_nodes - n)]),
                dtype='uint32'), time=relate_ts.tables.nodes.time)
        relate_ts_fixed = table_collection.tree_sequence()

        ts.dump('true_ts_' + str(rep) + '.trees')
        dated_ts.dump('dated_ts_' + str(rep) + '.trees')
        dated_inferred_ts_mut.dump('dated_inferred_ts_' + str(rep) + '.trees')
        relate_ts_fixed.dump('relate_ts_' + str(rep) + '.trees')
        compare_dict = compare_muts(n, ts, dated_ts,
                                    dated_inferred_ts_mut, ages)
        compare_df_master = pd.concat([compare_df_master, compare_dict])
        compare_df_master.to_csv("compare_df")


def compare_muts(n, ts, dated_ts, dated_inferred_ts_mut, ages):
    keep_ages = ages[(ages["Clock"] == "J") & (ages["Filtered"] == 1)]
    keep_ages = keep_ages.set_index('MarkerID')
    relate_age = pd.read_csv("compare.mut", sep=';')
    relate_age_estimates = relate_age['age_end'] - relate_age['age_begin']

    compare_dict = compare_mutations(ts, dated_ts, dated_inferred_ts_mut,
                                     keep_ages, relate_age)
    return(compare_dict)


compare_df_master = test_accuracy(10)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', '-r', type=int,
                        default=10, help="number of replicates")
    parser.add_argument(
        "--length", '-l', type=int, default=1e5,
        help="Length of the sequence")
    parser.add_argument('--seed', '-s', type=int,
                        default=123, help="use a non-default RNG seed")
    parser.add_argument(
        "--processes", '-p', type=int, default=1,
        help="number of worker processes, e.g. 40")
    args = parser.parse_args()
    np.random.seed(args.seed)
    random_seeds = np.random.randint(1, 2 ** 32, size=args.replicates)


if __name__ == "__main__":
    main()
