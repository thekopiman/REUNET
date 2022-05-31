import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from datetime import datetime


class TrainModel():
    def __init__(self, dataloaders) -> None:

        self.dataloaders = dataloaders

    def calc_loss_dense(self, pred, target, metrics):
        criterion = nn.MSELoss()
        loss = criterion(pred, target)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss

    def calc_loss_sparse(self, pred, target, samples, metrics, num_samples):
        criterion = nn.MSELoss()
        loss = criterion(samples*pred, samples*target)*(256**2)/num_samples
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss

    def print_metrics(self, metrics, epoch_samples, phase):
        outputs1 = []
        outputs2 = []
        for k in metrics.keys():
            outputs1.append("{}: {:4f}".format(
                k, metrics[k] / (epoch_samples*256**2)))

        print("{}: {}".format(phase, ", ".join(outputs1)))

    def train_model(self, model, optimizer, scheduler, num_epochs=50, WNetPhase="firstU", targetType="dense", num_samples=300, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), train_log=False, trainlog_path=None):
        # WNetPhase: traine first U and freez second ("firstU"), or vice verse ("secondU").
        # targetType: train against dense images ("dense") or sparse measurements ("sparse")
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10

        trainlog = {}

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("learning rate", param_group['lr'])
                        trainlog[epoch] = [param_group['lr']]
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0

                if targetType == "dense":
                    for inputs, targets in self.dataloaders[phase]:
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            [outputs1, outputs2] = model(inputs)
                            if WNetPhase == "firstU":
                                loss = self.calc_loss_dense(
                                    outputs1, targets, metrics)
                            else:
                                loss = self.calc_loss_dense(
                                    outputs2, targets, metrics)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        epoch_samples += inputs.size(0)
                elif targetType == "sparse":
                    for inputs, targets, samples in self.dataloaders[phase]:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        samples = samples.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            [outputs1, outputs2] = model(inputs)
                            if WNetPhase == "firstU":
                                loss = self.calc_loss_sparse(
                                    outputs1, targets, samples, metrics, num_samples)
                            else:
                                loss = self.calc_loss_sparse(
                                    outputs2, targets, samples, metrics, num_samples)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        epoch_samples += inputs.size(0)

                self.print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples
                trainlog[epoch].append(epoch_loss)

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            time_elapsed = time.time() - since
            trainlog[epoch].append(time_elapsed)
            print('{:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        print('Best val loss: {:4f}'.format(best_loss))
        if train_log:
            df = pd.DataFrame.from_dict(trainlog, orient='index', columns=[
                                        'learning_rate', 'train_loss', 'validation_loss', 'time'])
            if trainlog_path != None:
                df.to_csv(
                    f'{trainlog_path}/{datetime.now().strftime("%y%m%d_%H%M")}_trainlog_{WNetPhase}.csv')
            else:
                df.to_csv(
                    f'{datetime.now().strftime("%y%m%d_%H%M")}_trainlog_{WNetPhase}.csv')
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


class TestModel():
    def __init__(self) -> None:
        pass
