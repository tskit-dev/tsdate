import msprime
import tskit
import tsdate
from tqdm import tqdm

for i in tqdm(range(10)):
    ts = msprime.simulate(sample_size=100, Ne=10000, mutation_rate=1e-8, recombination_rate=1e-8, length=1e5)
    tsdate.age_inference(ts, progress=True)
