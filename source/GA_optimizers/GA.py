import numpy as np

def initialize(num_inds, num_params, domain=[0, 2], real_valued=False):
    lower_bound = domain[0]
    upper_bound = domain[1]
    if real_valued:
        pop = np.random.uniform(low=lower_bound, 
                                high=upper_bound, 
                                size=(num_inds, num_params))
    else:
        pop = np.random.randint(low=lower_bound, 
                                high=upper_bound+1, 
                                size=(num_inds, num_params))
    return pop

def evaluate(pop, func):
    f_pop = np.array(list(map(func, pop)))
    return f_pop

def max_gen_reached(gen, max_gen):
    return gen == max_gen

def pop_converge(pop):
    return len(np.unique(pop, axis=0)) == 1