import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from . import data as D
import numpy as np
import tqdm


def get_optimizer(net, algorithm, lr=0.001):
    if algorithm == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=lr)
    elif algorithm == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif algorithm == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=lr)
    elif algorithm == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=lr)
    elif algorithm == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif algorithm == 'adamax':
        optimizer = optim.Adamax(net.parameters())
    return optimizer


def run_deprecated(
        model: nn.Module,
        data: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        mode: str,
        scheduler,
        device,
        score_range=None
        ):
    if mode == 'train':
        model.train()
    elif mode == 'eval':
        model.eval()

    epoch_loss = 0
    data_len = 0
    preds = list()
    targets = list()
    feas = list()
    for x, mask, y in tqdm.tqdm(data, ncols=100):
        data_len += len(x)
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        res = model(x, mask)
        pred = res['output'].view(-1)
        fea = res['fea']
        preds.append(pred)
        feas.append(fea)
        targets.append(y)

        loss = criterion(pred, y)
        epoch_loss += len(x) * loss.item()
        if mode == 'train':
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

    # compute loss and qwk
    preds = torch.cat(preds).detach().cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    feas = torch.cat(feas).detach().cpu().numpy()
    if score_range:
        scores = D.recover_scores(preds, score_range)
        targets = D.recover_scores(targets, score_range)
    qwk = D.qwk(scores, targets)
    loss = epoch_loss/data_len
    return {
        'qwk': qwk, 'loss': loss, 'pred': preds, 'y': targets, 'fea': feas
    }


def run(
        model: nn.Module,
        data: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        mode: str,
        scheduler,
        device,
        score_range=None
        ):
    if mode == 'train':
        model.train()
    elif mode == 'eval':
        model.eval()

    epoch_loss = 0
    data_size = 0
    preds = list()
    feas = list()
    pred_att = list()

    normalized_targets = list()

    for dt in tqdm.tqdm(data, ncols=100):
        batch_size = len(dt[0])
        data_size += batch_size

        dt = [d.to(device) for d in dt]
        in_data = dt[:-1]
        y = dt[-1]

        optimizer.zero_grad()
        res = model(*in_data)
        pred = res['output'].view(-1)
        preds.append(pred)
        feas.append(res['fea'])
        if res['att'] is not None:
            if type(res['att']) == np.ndarray:
                pred_att.append(res['att'])
            else:
                pred_att.append(res['att'].detach().cpu().numpy())

        normalized_targets.append(y)

        loss = criterion(pred, y)
        epoch_loss += batch_size * loss.item()
        if mode == 'train':
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

    # compute loss and qwk
    normalized_preds = torch.cat(preds).detach().cpu().numpy()
    preds = normalized_preds
    normalized_targets = torch.cat(normalized_targets).cpu().numpy()
    targets = normalized_targets
    feas = torch.cat(feas).detach().cpu().numpy()
    if score_range is not None:
        preds = D.recover_scores(normalized_preds, score_range)
        targets = D.recover_scores(normalized_targets, score_range)
        qwk = D.qwk(preds, targets)
    else:
        qwk = -1
    loss = epoch_loss/data_size
    return {
        'qwk': qwk, 'loss': loss, 'fea': feas, 'att': pred_att,
        'pred': preds, 'normalized_preds': normalized_preds,
        'targets': targets, 'normalized_targets': normalized_targets
    }