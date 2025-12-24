import argparse
import os

import numpy as np

from train import *

agents = 5
w = np.array([[0.6, 0, 0, 0.4, 0],[0.2, 0.8, 0, 0, 0], [0.2, 0.1, 0.4, 0, 0.3], [0, 0, 0, 0.6, 0.4],[0, 0.1, 0.6, 0, 0.3]])

dataset = "cifar10"
bs = 32

def parse_args():
    ''' Function parses command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", default=0, type=int)
    parser.add_argument("-r","--run_num", default=0, type=int)
    parser.add_argument("-s", "--stratified", action='store_true')
    parser.add_argument("-e", "--epochs", default=800, type=int)
    parser.add_argument("-a", "--accuracy", default=1.0, type=float)
    parser.add_argument("-p", "--test_accuracy", default=1.0, type=float)
    return parser.parse_args()

args = parse_args()
epochs = args.epochs
max_accuracy = args.accuracy
max_test_accuracy = args.test_accuracy
cwd = os.getcwd()
results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

stratified = args.stratified
fname = os.path.join(results_path,f"{dataset}_e{epochs}_hom{stratified}_{args.test_num}.csv")


print(f"Test Num {args.test_num}, run num: {args.run_num}, {fname}")
if args.test_num == 0:
    DAdSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs,w=w, fname=fname, stratified=stratified, max_accuracy=max_accuracy, max_test_accuracy=max_test_accuracy)
elif args.test_num == 1:
    DLASTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, kappa=0.37, fname=fname, stratified=stratified, max_accuracy=max_accuracy, max_test_accuracy=max_test_accuracy)
elif args.test_num == 2:
    DAMSGradTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified, max_accuracy=max_accuracy, max_test_accuracy=max_test_accuracy)
elif args.test_num == 3:
    DAdaGradTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified, max_accuracy=max_accuracy, max_test_accuracy=max_test_accuracy)
elif args.test_num == 4:
    CDSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified, max_accuracy=max_accuracy, max_test_accuracy=max_test_accuracy)
elif args.test_num == 5:
    CDSGDPTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified, max_accuracy=max_accuracy, max_test_accuracy=max_test_accuracy)
elif args.test_num == 6:
    CDSGDNTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified, max_accuracy=max_accuracy, max_test_accuracy=max_test_accuracy)
