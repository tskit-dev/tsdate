import msprime
import tsinfer
import tskit

import tsdate

import numpy as np
from tqdm import tqdm
import pandas as pd
import subprocess
from sklearn.metrics import mean_squared_error, mean_squared_log_error

path_to_geva = "~/Documents/mcvean_group/tsinfer-with-branch-lengths/"


def generate_samples(ts, filename):
    """
    Generate a samples file from a simulated ts
    """

    assert ts.num_sites != 0
    sample_data = tsinfer.SampleData(path="tmp/"+filename + ".samples", sequence_length=ts.sequence_length)

    for v in ts.variants():
        sample_data.add_site(
            position=v.site.position, alleles=v.alleles,
            genotypes=v.genotypes)
    sample_data.finalise()

    return sample_data


def geva_age_estimate(file_name, Ne, mut_rate, rec_rate):
    """
    Perform GEVA age estimation on a given vcf
    """
    path = "/Users/anthonywohns/Documents/mcvean_group/age_inference/importance_sampling/" 
    subprocess.check_output([path + "geva/geva_v1beta", "--out", "tmp/" + file_name, "--rec", str(rec_rate), "--vcf", "tmp/" + file_name + ".vcf"])
    with open("tmp/"+file_name+".positions.txt","wb") as out:
        subprocess.call(["awk", "NR>3 {print last} {last = $3}", "tmp/"+file_name+".marker.txt"], stdout=out)
    try:
        subprocess.check_output([path + "geva/./geva_v1beta", "-i", "tmp/"+ file_name+".bin", "--positions", "tmp/"+file_name+".positions.txt","--hmm", path + "geva/hmm/hmm_initial_probs.txt", path + "geva/hmm/hmm_emission_probs.txt","--Ne", str(Ne), "--mut", str(mut_rate), "--maxConcordant","200","--maxDiscordant", "200","-o","tmp/"+file_name+"_estimation"])
    except subprocess.CalledProcessError as grepexc:
        print(grepexc.output)
        
    age_estimates = pd.read_csv("tmp/"+file_name+"_estimation.sites.txt", sep = " ")
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
    
    output_VCF = "tmp/"+filename+".vcf"
    with open(output_VCF, 'w') as vcf:
        vcf.write(header)

    df.to_csv(output_VCF, sep="\t", mode='a',index=False)

    return(geva_age_estimate(filename, Ne, mut_rate, rec_rate))

def get_mutation_child_parent(ts):
    mutation_child_parent = {mutation: None for mutation in range(ts.num_mutations)}
    for mutation in ts.mutations():
        focal_site = ts.site(mutation.site).position
        edge_num = np.intersect1d(np.argwhere(ts.tables.edges.child == mutation.node),
                       np.argwhere(np.logical_and(ts.tables.edges.left <= focal_site,
                                                  ts.tables.edges.right > focal_site)))
        mutation_child_parent[mutation.id] = (mutation.node, ts.edge(int(edge_num)).parent)
    return(mutation_child_parent)

for i in tqdm(range(1)):
    ts = msprime.simulate(sample_size=100, Ne=10000, mutation_rate=1e-8, recombination_rate=1e-8, length=1e5)
    sample_data = tsinfer.formats.SampleData.from_tree_sequence(ts)
    inferred_ts = tsinfer.infer(sample_data) 
    dated_ts = tsdate.age_inference(ts,theta=(4*10000*1e-8), rho=(4*10000*1e-8))
    dated_inferred_ts = tsdate.age_inference(inferred_ts, progress=True)
    print(mean_squared_log_error(dated_ts.tables.nodes.time, ts.tables.nodes.time)) 


for i in tqdm(range(0)):
    ts = msprime.simulate(sample_size=100, Ne=10000, mutation_rate=1e-7, recombination_rate=1e-8, length=1e5)
    sample_data = tsinfer.formats.SampleData.from_tree_sequence(ts)
    inferred_ts = tsinfer.infer(sample_data) 
    dated_ts = tsdate.age_inference(ts,theta=(4*10000*1e-7), rho=(4*10000*1e-8))
    dated_inferred_ts = tsdate.age_inference(inferred_ts, progress=True)
    print(mean_squared_log_error(dated_ts.tables.nodes.time, ts.tables.nodes.time)) 


samples = generate_samples(ts, 'testing')
ages = samplesdata_to_ages(samples, Ne=10000, length=1e5, mut_rate=1e-8, rec_rate=1e-8, filename=str("test"))

path = "/Users/anthonywohns/Documents/mcvean_group/software/relate_v1.0.13_MacOSX/"

def run_relate(ts, path):
    subprocess.check_output([path + "bin/RelateFileFormats", "--mode", "ConvertFromVcf", "--haps", path + "age_compare/compare.haps", "--sample", path + "age_compare/compare.sample", "-i", "tmp/test"])
    subprocess.check_output([path + "bin/Relate", "--mode", "All", "-m", "1e-8", "-N", "20000", "--haps", path + "age_compare/compare.haps", "--sample", path + "age_compare/compare.sample", "--seed", "1", "-o", "compare", "--map", path + "genetic_map.txt"])

run_relate(ts, path)

mutation_child_parent_ts = get_mutation_child_parent(ts)

keep_ages = ages[(ages["Clock"] == "J") & (ages["Filtered"] == 1)]
keep_ages = keep_ages.set_index('MarkerID')


relate_age = pd.read_csv("compare.mut", sep=';')
relate_age_estimates = relate_age['age_end'] - relate_age['age_begin'] 

truth = list()
geva = list()
relate = list()

for index, row in keep_ages.iterrows():
    (child, parent) = mutation_child_parent_ts[index]
    child_age = ts.node(ts.mutation(index).node).time
    parent_age = ts.node(parent).time
    true_age = (parent_age + child_age)/2
    truth.append(true_age)
    geva.append(row['PostMean']) 

for index, row in relate_age.iterrows():
    relate.append(row['age_end'] - row['age_begin']) 

print(mean_squared_log_error(truth, geva))
print(mean_squared_log_error(truth, relate))

np.savetxt('tmp/real_times', truth)
