from parent_trainer import *

class DAdSGDTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DAdSGD
        self.opt_name="DAdSGD"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch, dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25
        
        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, grad_diff, param_diff = {}, {}, {}, {}
            

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()
 
                self.prev_agent_optimizers[i].zero_grad()
                prev_predicted_label = self.prev_agent_models[i](inputs)
                prev_loss[i] = self.criterion(prev_predicted_label, labels)
                prev_loss[i].backward()

                grad_diff[i], param_diff[i] = self.agent_optimizers[i].compute_dif_norms(self.prev_agent_optimizers[i])

                if torch.cuda.device_count() > 1:
                    new_mod_state_dict = OrderedDict()
                    
                    for k, v in self.agent_models[i].state_dict().items():
                        new_mod_state_dict[k[7:]] = v
                    self.prev_agent_models[i].load_state_dict(new_mod_state_dict)
                else:
                    self.prev_agent_models[i].load_state_dict(self.agent_models[i].state_dict())


                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()
            
            for i in range(self.agents):
                self.agent_optimizers[i].set_norms(grad_diff[i], param_diff[i])
                self.agent_optimizers[i].step(self.running_iteration, vars=vars)
            
            if idx % log_interval == 0 and idx > 0 and epoch % self.print_freq == 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc

class DLASTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DLAS
        self.opt_name="DLAS"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch, dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25
        
        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, grad_diff, param_diff, lambdas = {}, {}, {}, {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                self.prev_agent_optimizers[i].zero_grad()
                prev_predicted_label = self.prev_agent_models[i](inputs)
                prev_loss[i] = self.criterion(prev_predicted_label, labels)
                prev_loss[i].backward()
                lambdas[i] = self.agent_optimizers[i].collect_lambda()
                grad_diff[i], param_diff[i] = self.agent_optimizers[i].compute_dif_norms(self.prev_agent_optimizers[i])

                if torch.cuda.device_count() > 1:
                    new_mod_state_dict = OrderedDict()
                    
                    for k, v in self.agent_models[i].state_dict().items():
                        new_mod_state_dict[k[7:]] = v
                    self.prev_agent_models[i].load_state_dict(new_mod_state_dict)
                else:
                    self.prev_agent_models[i].load_state_dict(self.agent_models[i].state_dict())

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()
            
            for i in range(self.agents):
                self.agent_optimizers[i].set_norms(grad_diff[i], param_diff[i])
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, lambdas=lambdas)

            if idx % log_interval == 0 and idx > 0 and epoch % self.print_freq == 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc

class CDSGDTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = CDSGD
        self.opt_name="CDSGD"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch, dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25
        
        loss = {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads = {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()


                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()
            
            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars)
            
            if idx % log_interval == 0 and idx > 0 and epoch % self.print_freq == 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc

class CDSGDPTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = CDSGDP
        self.opt_name="CDSGD-P"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch, dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25
        
        loss = {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads = {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()
            
            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars)
            
            if idx % log_interval == 0 and idx > 0 and epoch % self.print_freq == 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc

class CDSGDNTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = CDSGDN
        self.opt_name="CDSGD-N"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch, dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25
        
        loss = {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, grad_diff, param_diff, lambdas, all_y_vecs, old_grads = {}, {}, {}, {}, {}, {},{}
            old_y, u_tilde_5 = {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()
            
            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars)
            
            if idx % log_interval == 0 and idx > 0 and epoch % self.print_freq == 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc

class DAMSGradTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DAMSGrad
        self.opt_name="DAMSGrad"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch, dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25
        
        loss = {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, u_tilde_5 = {}, {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                if self.running_iteration > 0:
                    u_tilde_5[i] = self.agent_optimizers[i].collect_u()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)
                tot_loss += loss[i].item()
            
            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, u_tilde_5_all=u_tilde_5)

            if idx % log_interval == 0 and idx > 0 and epoch % self.print_freq == 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc

class DAdaGradTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DAdaGrad
        self.opt_name="DAdaGrad"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch, dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25
        
        loss = {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, u_tilde_5 = {}, {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                if self.running_iteration > 0:
                    u_tilde_5[i] = self.agent_optimizers[i].collect_u()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()
            
            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, u_tilde_5_all=u_tilde_5)

            
            if idx % log_interval == 0 and idx > 0 and epoch % self.print_freq == 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc
