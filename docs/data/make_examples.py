import msprime


def basic_example():
    ts = msprime.sim_ancestry(
        10, population_size=100, sequence_length=1e6, random_seed=123
    )
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=4321)
    ts.dump("basic_example.trees")


if __name__ == "__main__":
    basic_example()
