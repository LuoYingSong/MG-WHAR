import os
import random
import shutil
import time
from datetime import timedelta
import sys

import numpy as np
import torch
from sklearn import metrics as metrics
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader

from util.util import _init_fn, device, init_weights, paint, AverageMeter, Logger,makedir
from util.dataset import SensorDataset


def model_train(model, dataset, dataset_val, args, verbose=False):
    """
    Train model for a number of epochs.

    :param model: A pytorch model
    :param dataset: A SensorDataset containing the data to be used for training the model.
    :param dataset_val: A SensorDataset containing the data to be used for validation of the model.
    :param args: A dict containing config options for the training.
    Required keys:
                    'batch_size': int, number of windows to process in each batch (default 256)
                    'optimizer': str, optimizer function to use. Options: 'Adam' or 'RMSProp'. Default 'Adam'.
                    'lr': float, maximum initial learning rate. Default 0.001.
                    'lr_step': int, interval at which to decrease the learning rate. Default 10.
                    'lr_decay': float, factor by which to  decay the learning rate. Default 0.9.
                    'init_weights': str, How to initialize weights. Options 'orthogonal' or None. Default 'orthogonal'.
                    'epochs': int, Total number of epochs to train the model for. Default 300.
                    'print_freq': int, How often to print loss during each epoch if verbose=True. Default 100.

    :param verbose:
    :return:
    """
    if verbose:
        print(paint("Running HAR training loop ..."))

    loader = DataLoader(dataset, args['batch_size'], True, pin_memory=True, worker_init_fn=_init_fn)
    loader_val = DataLoader(
        dataset_val, args['batch_size'], True, pin_memory=True, worker_init_fn=_init_fn)

    criterion = nn.CrossEntropyLoss(reduction="mean").to(device=device)

    params = filter(lambda p: p.requires_grad, model.parameters())

    if args['optimizer'] == "Adam":
        optimizer = optim.Adam(params, lr=args['lr'])
    elif args['optimizer'] == "RMSprop":
        optimizer = optim.RMSprop(params, lr=args['lr'])

    if args['lr_step'] > 0:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args['lr_step'], gamma=args['lr_decay']
        )

    if args['init_weights'] == "orthogonal":
        if verbose:
            print(paint("[-] Initializing weights (orthogonal)..."))
        model.apply(init_weights)

    metric_best = 0.0
    start_time = time.time()
    loss, acc, fm, fw = eval_one_epoch(
        model, loader, criterion, args
    )
    loss_val, acc_val, fm_val, fw_val = eval_one_epoch(
        model, loader_val, criterion, args
    )
    n_epochs = args['epochs']
    acc_list = [acc_val]
    test_losses = [loss]
    train_losses = [loss_val]
    for epoch in range(n_epochs):
        if verbose:
            print("--" * 50)
            print("[-] Learning rate: ", optimizer.param_groups[0]["lr"])
        train_one_epoch(model, loader, criterion, optimizer, args, verbose)
        loss, acc, fm, fw = eval_one_epoch(
            model, loader, criterion, args
        )
        start_inf = time.time()
        loss_val, acc_val, fm_val, fw_val = eval_one_epoch(
            model, loader_val, criterion, args
        )
        inf_time = round(time.time() - start_inf)
        train_losses.append(loss)
        test_losses.append(loss_val)
        acc_list.append(acc_val)
        if verbose:
            print(
                paint(
                    f"[-] Epoch {epoch}/{args['epochs']}"
                    f"\tTrain loss: {loss:.2f} \tacc: {100 * acc:.2f}(%)\tfm: {100 * fm:.2f}(%)\tfw: {100 * fw:.2f}"
                    f"(%)\t"
                )
            )

            print(
                paint(
                    f"[-] Epoch {epoch}/{args['epochs']}"
                    f"\tVal loss: {loss_val:.2f} \tacc: {100 * acc_val:.2f}(%)\tfm: {100 * fm_val:.2f}(%)"
                    f"\tfw: {100 * fw_val:.2f}(%)"
                )
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }

        metric = fm_val
        if metric >= metric_best:
            if verbose:
                print(
                    paint(f"[*] Saving checkpoint... ({metric_best}->{metric})", "blue"))
            metric_best = metric
            torch.save(
                checkpoint, os.path.join(
                    model.path_checkpoints, "checkpoint_best.pth")
            )

        if epoch % 5 == 0:
            torch.save(
                checkpoint,
                os.path.join(model.path_checkpoints,
                             f"checkpoint_{epoch}.pth"),
            )

        if args['lr_step'] > 0:
            scheduler.step()
    np.savetxt(os.path.join(model.path_logs, 'train_loss'), np.array(train_losses), fmt='%f')
    np.savetxt(os.path.join(model.path_logs, 'test_loss'), np.array(test_losses), fmt='%f')
    np.savetxt(os.path.join(model.path_logs, 'test_acc'), np.array(acc_list), fmt='%f')
    elapsed = round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))
    if verbose:
        print(paint(f"Finished HAR training loop (h:m:s): {elapsed}"))
        print(paint("--" * 50, "blue"))


def train_one_epoch(model, loader, criterion, optimizer, args, verbose=False):
    losses = AverageMeter("Loss")
    model.train()
    for batch_idx, (data, target, idx) in enumerate(loader):

        data = data.to(device=device)
        target = target.view(-1).to(device=device)

        logits = model(data)
        loss = criterion(logits, target)

        losses.update(loss.item(), data.shape[0])
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if verbose:
            if batch_idx % args['print_freq'] == 0:
                print(
                    f"[-] Batch {batch_idx}/{len(loader)}\t Loss: {str(losses)}")


def eval_one_epoch(model, loader, criterion, args, to_file=False):
    losses = AverageMeter("Loss")
    y_true, y_pred = [], []
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target, idx) in enumerate(loader):
            data = data.to(device=device)
            target = target.to(device=device)

            logits = model(data)
            loss = criterion(logits, target.view(-1))
            losses.update(loss.item(), data.shape[0])

            probabilities = nn.Softmax(dim=1)(logits)
            _, predictions = torch.max(probabilities, 1)

            y_pred.append(predictions.cpu().numpy().reshape(-1))
            y_true.append(target.cpu().numpy().reshape(-1))

    # append invalid samples at the beginning of the test sequence
    if loader.dataset.prefix == "test":
        ws = data.shape[1] - 1
        samples_invalid = [y_true[0][0]] * ws
        y_true.append(samples_invalid)
        y_pred.append(samples_invalid)

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)

    acc = metrics.accuracy_score(y_true, y_pred)
    if to_file:
        np.savetxt(os.path.join(model.path_logs, 'true'), y_true, fmt='%d')
        np.savetxt(os.path.join(model.path_logs, 'y_pred'), y_pred, fmt='%d')
    fm = metrics.f1_score(y_true, y_pred, average="macro")
    fw = metrics.f1_score(y_true, y_pred, average="weighted")

    return losses.avg, acc, fm, fw


def model_eval(model, dataset_test, args, return_results):
    print(paint("Running HAR evaluation loop ..."))

    loader_test = DataLoader(
        dataset_test, args['batch_size'], False, pin_memory=True, worker_init_fn=_init_fn)

    criterion = nn.CrossEntropyLoss(reduction="mean").to(device=device)

    print("[-] Loading checkpoint ...")

    path_checkpoint = os.path.join(
        model.path_checkpoints, "checkpoint_best.pth")

    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion.load_state_dict(checkpoint["criterion_state_dict"])

    start_time = time.time()

    loss_test, acc_test, fm_test, fw_test = eval_one_epoch(
        model, loader_test, criterion, args, True
    )

    print(
        paint(
            f"[-] Test loss: {loss_test:.2f}"
            f"\tacc: {100 * acc_test:.2f}(%)\tfm: {100 * fm_test:.2f}(%)\tfw: {100 * fw_test:.2f}(%)"
        )
    )

    elapsed = time.time() - start_time
    # elapsed = str(timedelta(seconds=elapsed))
    # print(paint(f"[Finished HAR evaluation loop (h:m:s): {elapsed}"))

    if return_results:
        return acc_test, fm_test, fw_test, elapsed, loss_test


def test(model, length, dataset_path, log_path, config_train, test_config, mode='restart', use_nni = True, show_test=False, word=None, topic=None):
    assert mode in ('restart', 'recovery', 'test')
    paint('code running on the :' + device)
    model_name = model.__class__.__name__
    log_path = os.path.join(log_path, model_name)
    makedir(log_path)
    sys.stdout = Logger(stream=sys.stdout, filename=os.path.join(log_path, '{}.txt'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))))
    train_set = SensorDataset('opportunity', length, length // 2, dataset_path, 'train')
    test_set = SensorDataset('opportunity', length, length // 2, dataset_path, 'test')
    valid_set = SensorDataset('opportunity', length, length // 2, dataset_path, 'val')
    if mode == 'recovery':
        config_train['init_weights'] = ''
        criterion = nn.CrossEntropyLoss(reduction="mean").to(device=device)
        path_checkpoint = os.path.join(
        model.path_checkpoints, "checkpoint_best.pth")
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        criterion.load_state_dict(checkpoint["criterion_state_dict"])
    if mode != 'test':
        if show_test:
            test_set = valid_set
        model_train(model, train_set, test_set, config_train, True)
    test_config['dataset'] = valid_set
    acc_test, fm_test, fw_test, elapsed, loss = model_eval(model, valid_set, test_config, True)
    with open(os.path.join(log_path, 'result.txt'), 'a') as f:
        f.write(f'{word} {topic} {acc_test}, {fm_test}, {fw_test}, {elapsed} {loss}\n')
    return acc_test, fm_test, fw_test, elapsed, loss
