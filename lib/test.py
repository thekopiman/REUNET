import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader


def calc_loss_test(pred1, pred2, target, metrics, error="MSE"):
    criterion = nn.MSELoss()
    if error == "MSE":
        loss1 = criterion(pred1, target)
        loss2 = criterion(pred2, target)
    else:
        loss1 = criterion(pred1, target)/criterion(target, 0*target)
        loss2 = criterion(pred2, target)/criterion(target, 0*target)
    metrics['loss first U'] += loss1.data.cpu().numpy() * target.size(0)
    metrics['loss second U'] += loss2.data.cpu().numpy() * target.size(0)

    return [loss1, loss2]


def print_metrics_test(metrics, epoch_samples, error):
    outputs = []
    if error == "MSE":
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(
                k, metrics[k] / (epoch_samples*256**2)))
    else:
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format("Test"+" "+error, ", ".join(outputs)))


def test_loss(model, Radio_test, batch_size, error="MSE", dataset="coarse"):
    # dataset is "coarse" or "fine".
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    model.eval()   # Set model to evaluate mode
    metrics = defaultdict(float)
    epoch_samples = 0
    if dataset == "coarse":
        for inputs, targets in DataLoader(Radio_test, batch_size=batch_size, shuffle=True, num_workers=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # do not track history if only in train
            with torch.set_grad_enabled(False):
                [outputs1, outputs2] = model(inputs)
                [loss1, loss2] = calc_loss_test(
                    outputs1, outputs2, targets, metrics, error)
                epoch_samples += inputs.size(0)
    elif dataset == "fine":
        for inputs, targets, samples in DataLoader(Radio_test, batch_size=batch_size, shuffle=True, num_workers=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # do not track history if only in train
            with torch.set_grad_enabled(False):
                [outputs1, outputs2] = model(inputs)
                [loss1, loss2] = calc_loss_test(
                    outputs1, outputs2, targets, metrics, error)
                epoch_samples += inputs.size(0)
    print_metrics_test(metrics, epoch_samples, error)
    #test_loss1 = metrics['loss U'] / epoch_samples
    #test_loss2 = metrics['loss W'] / epoch_samples
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
