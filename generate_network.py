import itertools
from causalgraphicalmodels import CausalGraphicalModel
import numpy as np
import pandas as pd
import random


def create_network_from_file(file_name):
    with open(file_name) as f:
        random_seed = f.readline().strip().split("random_seed: ")[1]
        random_seed = int(random_seed) if random_seed != "no_seed" else 42

        random.seed(random_seed)
        np.random.seed(random_seed)

        variable_names = f.readline().strip().split("variable_names: ")[1].split(" ")
        nsamp = int(f.readline().strip().split("number_of_samples: ")[1])
        noise_mean, noise_variance = map(float, f.readline().strip().split("normal_noise: ")[1].split(" "))

        df = pd.DataFrame(np.random.normal(noise_mean, noise_variance, size=(nsamp, len(variable_names))),
                          columns=variable_names)

        number_of_generating_functions = int(f.readline().strip().split("number_of_generating_functions: ")[1])
        for i in range(number_of_generating_functions):
            var_name, min_v, max_v = f.readline().strip().split(" ")
            df[var_name] = np.random.uniform(float(min_v), float(max_v), nsamp)
        rest_of_dependencies = map(str.strip, f.readlines())
        edges = []
        noise_and_generating = df.copy()
        for dep in rest_of_dependencies:
            end_node, expression = dep.split(" = ")
            start_nodes = [variable_name for variable_name in variable_names if variable_name in expression]
            edges_for_dep = list(itertools.product(start_nodes, end_node))
            # augmentation = " = " + dep.split(" = ")[0]  + " + "
            df[dep.split(" = ")[0]] += df.eval(dep.split(" = ")[1])
            # print(dep.replace(" = ", augmentation))
            edges.extend(edges_for_dep)

    Bayesian_network = CausalGraphicalModel(nodes=variable_names, edges=edges)
    return df, Bayesian_network, noise_and_generating