import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from . import data as D
from . import utils as U
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


def kappa_loss(p, y, n_classes=5, eps=1e-10):
    """
    QWK loss function as described in https://arxiv.org/pdf/1612.00775.pdf
    Arguments:
        p: a tensor with probability predictions, [batch_size, n_classes],
        y, a tensor with one-hot encoded class labels, [batch_size, n_classes]
    Returns:
        QWK loss
    """

    W = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = (i-j)**2

    W = torch.from_numpy(W.astype(np.float32)).to(device)
    O = torch.matmul(y.t(), p)
    E = torch.matmul(y.sum(dim=0).view(-1, 1), p.sum(dim=0).view(1, -1)) / O.sum()

    return (W*O).sum() / ((W*E).sum() + eps)


def run(
        model: nn.Module,
        data: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        mode: str,
        scheduler,
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
    for x, x_len, y in tqdm.tqdm(data):
        data_len += len(x)

        optimizer.zero_grad()
        pred = model(x, x_len).view(-1)
        preds.append(pred)
        targets.append(y)

        loss = criterion(pred, y)
        epoch_loss += len(x) * loss.item()
        if mode == 'train':
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

    # compute loss and qwk
    preds = torch.cat(preds).detach().numpy()
    if score_range:
        preds = D.recover_scores(preds, score_range)
    targets = torch.cat(targets).numpy()
    qwk = D.qwk(preds, targets)
    loss = epoch_loss/data_len
    return epoch_loss / data_len, qwk
