from collections import OrderedDict
import copy
import csv
from random import shuffle, sample
from time import perf_counter
import warnings
import sys # use exit() to quit on accuracy threshold instead of # of epochs

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from models import *
from ops import DAdSGD, DLAS, CDSGD, CDSGDP, CDSGDN, DAMSGrad, DAdaGrad

warnings.filterwarnings("ignore")

class DTrainer:
    def __init__(self,
                dataset="cifar10",
                epochs=100,
                batch_size=32,
                lr=0.02,
                workers=4,
                agents=5,
                num=0.5,
                kmult=0.0,
                exp=0.7,
                w=None,
                kappa=0.9,
                fname=None, # always given by main.py; name of file to save CSV output
                max_accuracy=1,
                max_test_accuracy=1,
                epoch_print_freq=2,
                stratified=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_iterations = []
        self.test_iterations = []
        self.lr_logs = {}
        self.lambda_logs = {}
        self.loss_list = []

        self.print_freq = epoch_print_freq
        ''' constant after being given by user (2 by default);
        controls how often info is printed to terminal'''
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.workers = workers
        self.agents = agents
        self.num = num
        self.kmult = kmult
        self.exp = exp
        self.kappa = kappa
        self.fname = fname
        self.max_accuracy = max_accuracy
        self.max_test_accuracy = max_test_accuracy
        self.stratified = stratified
        self.load_data()
        self.w = w
        self.criterion = torch.nn.CrossEntropyLoss()
        self.agent_setup()

    def _log(self, accuracy):
        '''
        Helper function to log accuracy values
        Appends values to list that will be written to file with _save
        '''
        self.train_accuracy.append(accuracy)
        self.train_iterations.append(self.running_iteration)

    def _save(self):
        '''
        method to save accuracy values
        only needs to be called once, once training is done
        '''
        with open(self.fname, mode='a') as csv_file: # append mode
            file = csv.writer(csv_file, lineterminator = '\n')
            file.writerow([f"{self.opt_name}, {self.num}, {self.kmult}, {self.batch_size}, {self.epochs}"])
            file.writerow(self.train_iterations)
            file.writerow(self.train_accuracy)
            file.writerow(self.test_iterations)
            file.writerow(self.test_accuracy)
            file.writerow(self.loss_list) # this and four rows above should be lists with values for every print_freq epochs in order
            file.writerow(["ETA"])
            for i in range(self.agents):
                file.writerow(self.lr_logs[i])
            if self.opt_name == "DLAS":
                file.writerow(["LAMBDA"])
                for i in range(self.agents):
                    file.writerow(self.lambda_logs[i])
            file.writerow([])

    def load_data(self):
        print("==> Loading Data")
        self.train_loader = {}
        self.test_loader = {}

        if self.dataset == 'cifar10':
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            self.class_num = 10
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        elif self.dataset == "mnist":
            transform_train = transforms.Compose([transforms.ToTensor(),])
            transform_test = transforms.Compose([transforms.ToTensor(),])

            self.class_num = 10
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        else:
            raise ValueError(f'{self.dataset} is not supported')

        if self.stratified:
            train_len, test_len = int(len(trainset)), int(len(testset))

            temp_train = torch.utils.data.random_split(trainset, [int(train_len//self.agents)]*self.agents)

            for i in range(self.agents):
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train[i], batch_size=self.batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
        else:
            train_len, test_len = int(len(trainset)), int(len(testset))
            idxs = {}
            for i in range(0, 10, 2):
                arr = np.array(trainset.targets, dtype=int)
                idxs[int(i/2)] = list(np.where(arr == i)[0]) + list(np.where(arr == i+1)[0])
                shuffle(idxs[int(i/2)])

            percent_main = 0.5
            percent_else = (1 - percent_main) / (self.agents-1)
            main_samp_num = int(percent_main * len(idxs[0]))
            sec_samp_num = int(percent_else * len(idxs[0]))

            for i in range(self.agents):
                agent_idxs = []
                for j in range(self.agents):
                    if i == j:
                        agent_idxs.extend(sample(idxs[j], main_samp_num))
                    else:
                        agent_idxs.extend(sample(idxs[j], sec_samp_num))
                    idxs[j] = list(filter(lambda x: x not in agent_idxs, idxs[j]))
                temp_train = copy.deepcopy(trainset)
                temp_train.targets = [temp_train.targets[i] for i in agent_idxs]
                temp_train.data = [temp_train.data[i] for i in agent_idxs]
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train, batch_size=self.batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    def agent_setup(self):
        for i in range(self.agents):
            self.lr_logs[i] = []
            self.lambda_logs[i] = []

        self.agent_models = {}
        self.prev_agent_models = {}
        self.agent_optimizers = {}
        self.prev_agent_optimizers = {}

        if self.dataset == 'cifar10':
            model = CifarCNN()

        elif self.dataset == "imagenet":
            raise ValueError("ImageNet Not Supported: Low Computing Power")

        elif self.dataset == "mnist":
            model = MnistCNN()

        for i in range(self.agents):
            if i == 0:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = torch.nn.DataParallel(model)
                else:
                    self.agent_models[i] = model

            else:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = copy.deepcopy(self.agent_models[0])
                else:
                    self.agent_models[i] = copy.deepcopy(model)

            self.agent_models[i].to(self.device)
            self.agent_models[i].train()

            if self.opt_name == "DAdSGD" or self.opt_name == "DLAS":
                self.prev_agent_models[i] = copy.deepcopy(model)
                self.prev_agent_models[i].to(self.device)
                self.prev_agent_models[i].train()
                self.prev_agent_optimizers[i] = self.opt(
                                params=self.prev_agent_models[i].parameters(),
                                idx=i,
                                w=self.w,
                                agents=self.agents,
                                lr=self.lr,
                                num=self.num,
                                kmult=self.kmult,
                                name=self.opt_name,
                                device=self.device,
                                kappa=self.kappa,
                                stratified=self.stratified
                            )

            self.agent_optimizers[i] = self.opt(
                            params=self.agent_models[i].parameters(),
                            idx=i,
                            w=self.w,
                            agents=self.agents,
                            lr=self.lr,
                            num=self.num,
                            kmult=self.kmult,
                            name=self.opt_name,
                            device=self.device,
                            kappa=self.kappa,
                            stratified=self.stratified
                        )

    def eval(self, dataloader):
        total_acc, total_count = 0, 0

        with torch.no_grad():

            for i in range(self.agents):
                self.agent_models[i].eval()

                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    predicted_label = self.agent_models[i](inputs)

                    total_acc += (predicted_label.argmax(1) == labels).sum().item()
                    total_count += labels.size(0)

        self.test_iterations.append(self.running_iteration)
        self.test_accuracy.append(total_acc/total_count)

        return total_acc/total_count

    def it_logger(self, total_acc, total_count, epoch, log_interval, tot_loss, start_time):
        self._log(total_acc/total_count)
        acc = total_acc/total_count
        t_acc = self.eval(self.test_loader)
        if acc < self.max_accuracy and t_acc < self.max_test_accuracy:
            for i in range(self.agents):
                self.lr_logs[i].append(self.agent_optimizers[i].collect_params(lr=True))
                if self.opt_name == "DLAS":
                    self.lambda_logs[i].append(self.agent_optimizers[i].collect_lambda())

            ss = self.lr_logs[0][-1] if self.opt_name != "DLAS" else self.lambda_logs[0][-1]
            print(
                f"Epoch: {epoch+1}, Iteration: {self.running_iteration}, "+
                f"Accuracy: {acc:.4f}, "+
                f"Test Accuracy: {t_acc:.4f}, " +
                f"Loss: {tot_loss/(self.agents * log_interval):.4f}, "+
                f"ss: {ss:.5f}, "+
                f"Time taken: {perf_counter()-start_time:.4f}"
            )

            self.loss_list.append(tot_loss/(self.agents * log_interval))
        else:
            print(f"Reached specified accuracy threshold: accuracy {acc:.4f}, test accuracy {t_acc:.4f}.  Took {epoch} epochs.")
            #sys.stdout.flush()
            self._save() # save output in csv file
            sys.exit(0) # reached desired accuracy
    def trainer(self):
        if self.opt_name == "DAdSGD" or self.opt_name == "DLAS":
            print(f"==> Starting Training for {self.opt_name}, {self.epochs} epochs and {self.agents} agents on the {self.dataset} dataset, via {self.device}")
        else:
            print(f"==> Starting Training for {self.opt_name}, {self.epochs} epochs and {self.agents} agents on the {self.dataset} dataset, via {self.device}" +
                  f" for {self.num}, {self.kmult}")
        for i in range(self.agents):
            self.test_accuracy = []
            self.train_accuracy = []

        for i in range(self.epochs):
            self.epoch_iterations(i, self.train_loader)
