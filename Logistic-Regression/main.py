import argparse
import os
import sys

import numpy as np

from trainer_classes import *
from parent_trainer import load_data, plot_all_losses

dataset = "mushrooms"
cwd = os.getcwd()
data_path = os.path.join(cwd, "mushrooms")

def parse_args():
    ''' Function parses command line arguments '''

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stratified", action='store_true')
    parser.add_argument("-k", "--kappa", default=0.4, type=float)
    parser.add_argument("-i", "--iterations", default=5000, type=int)
    parser.add_argument("-g","--print_gap", default=200, type=int)
    return parser.parse_args()

args = parse_args()
stratified = args.stratified
label = "homogeneous" if stratified == True else "heterogeneous"

agents = 5

data = load_data(stratified=stratified, data_path=data_path, agents=agents)
agent_matrix = np.array([[0.6, 0, 0, 0.4, 0],[0.2, 0.8, 0, 0, 0], [0.2, 0.1, 0.4, 0, 0.3], [0, 0, 0, 0.6, 0.4],[0, 0.1, 0.6, 0, 0.3]])

# iterations = 5000
iterations = args.iterations
decay = 0.9
kap = args.kappa
print_gap = args.print_gap
# kap = 0.4

dadgd = DAdGD(agent_matrix=agent_matrix, iterations=iterations, data=data, print_gap=print_gap)
dadgd.train()

doas = DOAS(kap=kap, agent_matrix=agent_matrix, iterations=iterations, data=data, print_gap=print_gap)
doas.train()

cdgd = CDGD(agent_matrix=agent_matrix, iterations=iterations,data=data, print_gap=print_gap)
cdgd.train()

cdgdp = CDGDP(momentum_param=decay, agent_matrix=agent_matrix, iterations=iterations, data=data, print_gap=print_gap)
cdgdp.train()

cdgdn = CDGDN(momentum_param=decay, agent_matrix=agent_matrix, iterations=iterations, data=data, print_gap=print_gap)
cdgdn.train()

damsgrad = DAMSGrad(agent_matrix=agent_matrix, iterations=iterations, data=data, print_gap=print_gap)
damsgrad.train()

dadagrad = DAdaGrad(agent_matrix=agent_matrix, iterations=iterations, data=data, print_gap=print_gap)
dadagrad.train()

optimizers = [dadgd, doas, cdgd, cdgdp, cdgdn, damsgrad, dadagrad]
plot_all_losses(optimizers, skip=1)


results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

filename = os.path.join(results_path, f"{dataset}_{label}.csv")
for opt in optimizers:
    opt.save_data(filename)
