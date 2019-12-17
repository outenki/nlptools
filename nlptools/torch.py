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
