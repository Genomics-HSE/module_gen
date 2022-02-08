import os
import sys
from math import (exp, log)

import msprime
import numpy as np

N = 20

T_max = 20#400_000
coeff = np.log(T_max)/(N-1)

limits = [np.exp(coeff*i) for i in range(N)]
limits = [2_000*(np.exp(i*np.log(1+10*T_max)/N )-1) for i in range(N)]


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


L_HUMAN = 3_000_000 # int(float(sys.argv[3])) #30_000_000
RHO_HUMAN = 1.6*10e-9
MU_HUMAN = 1.25*10e-8

RHO_LIMIT = (1.6*10e-10, 1.6*10e-8)
MU_LIMIT = (1.25*10e-7, 1.25*10e-9)

NUMBER_OF_EVENTS_LIMITS = (1, 20)
LAMBDA_EXP = 20_000

POPULATION = 10_000
POPULATION_COEFF_LIMITS = (0.5, 1.5)

MIN_POPULATION_NUM = 1_000
MAX_POPULATION_NUM = 120_000

POPULATION_SIZE_LIMITS = (MIN_POPULATION_NUM, MAX_POPULATION_NUM)

lambda_exp = 500
coal_limits = .0001 #0.999

POPULATION_COEFF_LIMIT_COMPLEX = [1.0, 2.0]

SPECIAL_COEFF = 800 #1_000 #2_000


limits = [np.exp(coeff*i) for i in range(N)]
limits = [SPECIAL_COEFF*(np.exp(i*np.log(1+10*T_max)/N )-1) for i in range(N)]

def give_rho() -> float:
    return np.random.uniform(*RHO_LIMIT)


def give_mu() -> float:
    return np.random.uniform(*MU_LIMIT)


def give_random_coeff(mean=.128, var=.05) -> float:
    return np.random.beta(.1, .028)*.0128


def give_random_rho(base=RHO_HUMAN) -> float:
    return np.random.uniform(0.0001, 100, 1)[0]*base

def give_population_size() -> int:
    return int(np.random.uniform(*[10_000,29_000]))

def generate_demographic_events_complex(population: int = None, random_seed=42) -> 'msprime.Demography':
    np.random.seed(random_seed)
    if not population:
        population = give_population_size()
        
    p =[0]
    
    while p[-1] < 1/10_000:
    
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=population)

        last_population_size = population
        T = 0
        coal_probability = 0.0
        coal_probability_list = [] 
        non_coal_probability = 1.0
        timees = []
        while T < 420_000:
            t = np.random.exponential(lambda_exp)
            timees.append(t)
            T += t


            #last_population_size = max(last_population_size * np.random.uniform(*POPULATION_COEFF_LIMITS),
            #                           MIN_POPULATION_NUM)

            coeff = (np.random.uniform(*POPULATION_COEFF_LIMIT_COMPLEX))**(np.random.choice([-1, 1]))
            # print(last_population_size)
            last_population_size = min(max(last_population_size * coeff, MIN_POPULATION_NUM),MAX_POPULATION_NUM)

            demography.add_population_parameters_change(
                T, initial_size=last_population_size)

            coal_probability = non_coal_probability + t/last_population_size
            coal_probability_list.append(coal_probability)
            non_coal_probability = non_coal_probability + (-t/last_population_size)


        # check if demography is ok
        print(timees[:5])
        debug = demography.debug()
        def f(x):
            for (a, t) in zip(debug.population_size_history[0], debug.epoch_start_time):
                if x >= t:
                    return a
            return 0

        coeffs_ = []
        for (a,b) in zip(limits[:-1], limits[1:]):
            coeffs_.append((b-a)*1/(f(a) + f(b)/2)) 
        p = []
        for i,c in enumerate(coeffs_):
            p.append((1-np.exp(-c)) * np.exp(-np.sum(coeffs_[:i])))
    
        # if probbobility of last category is too low we generate new demography
        if p[-1] < 1/10_000:
            print(f"Bad demography:{p[:5], p[-1]}")
    #print(p)
    return demography

def generate_demographic_events(population: int = None) -> 'msprime.Demography':
    
    if not population:
        population = give_population_size()
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=population)

    number_of_events = np.random.randint(*NUMBER_OF_EVENTS_LIMITS)

    times = sorted(np.random.exponential(LAMBDA_EXP, size=number_of_events))

    last_population_size = population
    for t in times:
        last_population_size = max(last_population_size * np.random.uniform(*POPULATION_COEFF_LIMITS),
                                   MIN_POPULATION_NUM)
        demography.add_population_parameters_change(
            t, initial_size=last_population_size)

    return demography


def simple_split(time: float, N: int, split_const: int = 5000) -> int:
    return int(min(time//split_const, N-1))


def exponent_split(time: float, N: int) -> int:
    for i, limit in enumerate(limits):
        if limit > time:
            return i
    return N-1 

class DataGenerator():
    def __init__(self,
                 recombination_rate: float = RHO_HUMAN,
                 mutation_rate: float = MU_HUMAN,
                 demographic_events: list = None,
                 population: int = None,
                 number_intervals: int = N,
                 splitter=simple_split,  # maust be annotiede
                 num_replicates: int = 1,
                 lengt: int = L_HUMAN,
                 model: str = "hudson",
                 random_seed: int = 42,
                 mutation_random_seed: int = 42,
                 sample_size: int = 1,
                 ):

        self.sample_size = sample_size
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate
        self.num_replicates = num_replicates
        if not demographic_events:
            if not population:
                raise BaseException(
                    "Eiter demographic_events or population must be speciefied")
            demographic_events = msprime.Demography()
            demographic_events.add_population(
                name="A", initial_size=population)
        self.demographic_events = demographic_events
        self.splitter = splitter
        self.model = model
        self.len = lengt
        self.random_seed = random_seed
        self.mutation_random_seed = mutation_random_seed
        self.number_intervals = number_intervals
        self._data = None

    def run_simulation(self):
        """
        return generator(tskit.TreeSequence)
        function run the simulation with given parametrs
        """
        self._data = msprime.sim_ancestry(
            recombination_rate=self.recombination_rate,
            sequence_length=self.len,
            num_replicates=self.num_replicates,
            demography=self.demographic_events,
            model=self.model,
            random_seed=self.random_seed,
            samples=self.sample_size)
        return self._data

    def __iter__(self):
        return self

    def __next__(self):
        """
        return haplotype, recombination points and coalescent time
        """
        if self._data is None:
            self.run_simulation()

        try:
            tree = next(self._data)
        except StopIteration:
            raise StopIteration

        mutated_ts = msprime.sim_mutations(
            tree, rate=self.mutation_rate, random_seed=self.mutation_random_seed)  # random_seed
        
        self.mutation_random_seed += 1

        #times = [0]*self.len
        d_times = [0]*self.len
        mutations = [0]*self.len
        prior_dist = [0.0]*self.number_intervals

        for m in mutated_ts.mutations():
            mutations[int(m.site)] = 1

        for t in mutated_ts.aslist():
            interval = t.get_interval()
            left = interval.left
            right = interval.right
            time = t.get_total_branch_length()/2
            #times[int(left):int(right)] = [time]*int(right-left)
            d_times[int(left):int(right)] = [self.splitter(
                time, self.number_intervals)]*int(right-left)
            prior_dist[self.splitter(
                time, self.number_intervals)] += (int(right-left))/self.len

        return mutations, d_times, prior_dist


if __name__ == "__main__":

    num_model = int(sys.argv[6])
    x_path = os.path.join(sys.argv[1], "x")
    y_path = os.path.join(sys.argv[1], "y")
    pd_path = os.path.join(sys.argv[1], "PD")

    name = 0
    print(f'Path: {sys.argv[1]}')
    print(f"Num_model: {num_model}")
    print(f"Num replicates: {sys.argv[2]}")
    for j in range(num_model):
        generator = DataGenerator(
            demographic_events=generate_demographic_events_complex(),
            splitter=exponent_split,
            num_replicates= int(sys.argv[2])
        )
        generator.run_simulation()
        # return mutations, times, None, prior_dist, None
        for x, y, t in generator:
            print(name)
            x = np.array(x, dtype=np.int64)
            y = np.array(y)
            pd = np.array(t)

            np.save(x_path + "/" + str(name), x)
            np.save(y_path + "/" + str(name), y)
            np.save(pd_path + "/" + str(name), pd)
            name += 1

            
### New gnrators models ###

import typing as tp

def generate_demography_from_list(
    times: tp.List[float],
    sizes: tp.List[float],
    initial_size = None) -> msprime.Demography:
    
    if initial_size is None:
        initial_size = sizes[0]
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)
    start = 1 if len(times) != len(sizes) else 0
    for time, size  in zip(times, sizes[start:]):
        demography.add_population_parameters_change(                
            time=time,
            growth_rate=0,
            initial_size=size
        )
    return demography


def SNP_generator(
    segment_length,
    demography,
    recombination_rate,
    mutation_rate,
    num_replicates,
    seed=42
):  
    tree_sequence = msprime.simulate(
                length=segment_length,
                demography=demography,
                # population_configurations=population_configurations,
                # demographic_events=demographic_events,
                recombination_rate=recombination_rate,
                mutation_rate=mutation_rate,
                num_replicates=num_replicates,
                random_seed=seed)
    
    pos = []
    snp = []
    for i, rep in enumerate(tree_sequence):
                positions = [variant.site.position for variant in rep.variants()]
                positions = np.array(positions) - np.array([0] + positions[:-1])
                positions = positions.astype(int)
                pos.append(positions)
                SNPs = rep.genotype_matrix().T.astype(np.uint8)
                snp.append(SNPs)
    data = [[snp[i], pos[i]] for i in range(len(snp))]
    data = [np.vstack([d[1], d[0]]) for d in data]
    return data



class BaseGenomeExtractor:
    
    def __init__(self, generator):
        self.generator = generator
    
    def __call__(self):
        yield from self.generator
        
        
def generateATCG(generator, l=100, p = [.25, .25, .25 , .25]):
    genome = [g+g for g in np.random.choice(["A", "T", "C", "G"], size=l,p=p)]
    for var in generator.variants():
        genome[int(var.site.position)] = var.alleles[0] + var.alleles[1]
    return genome


def genome_matrix(generator, sample_size=4):
    # must be generator that return sasmples of size sample_size
    pass



### from the paper ###

def simulate_scenario(population_size, population_time, seed, num_replicates, mutation_rate, recombination_rate, 
                      segment_length, num_sample):
    
    demographic_events = [msprime.PopulationParametersChange(
                time=population_time[i],
                growth_rate=0,
                initial_size=population_size[i]) for i in range(1, len(population_time))]

    population_configurations = [msprime.PopulationConfiguration(
                sample_size=num_sample,
                initial_size=population_size[0])]

    tree_sequence = msprime.simulate(
                length=segment_length,
                population_configurations=population_configurations,
                demographic_events=demographic_events,
                recombination_rate=recombination_rate,
                mutation_rate=mutation_rate,
                num_replicates=num_replicates,
                random_seed=seed)
    pos = []
    snp = []
    for i, rep in enumerate(tree_sequence):
                positions = [variant.site.position for variant in rep.variants()]
                positions = np.array(positions) - np.array([0] + positions[:-1])
                positions = positions.astype(int)
                pos.append(positions)
                SNPs = rep.genotype_matrix().T.astype(np.uint8)
                snp.append(SNPs)
    data = [[snp[i], pos[i]] for i in range(len(snp))]
    data = [np.vstack([d[1], d[0]]) for d in data]
    return data


