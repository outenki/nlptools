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
        pred = res['pred'].view(-1)
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
        score_range=None,
        att_loss=False,
        ):
    '''
    Run training/test step for one epoch.
    Note: the length of <data> is not fixed.
        The last element of <data> will be used as target of prediction,
        while the others will be passed to the model as position arguments
    '''
    if mode == 'train':
        model.train()
    elif mode == 'eval':
        model.eval()

    epoch_loss = 0
    data_size = 0
    preds = list()
    # feas = list()
    pred_att = list()
    normalized_targets = list()

    keys_indices = list()
    keys_weights = list()

    for dt in tqdm.tqdm(data, ncols=100):
        batch_size = len(dt[0])
        data_size += batch_size

        dt = [d.to(device) for d in dt]
        in_data = dt[:-1]
        y = dt[-1]

        optimizer.zero_grad()
        res = model(*in_data)
        # pred = res['pred'].view(-1)
        if 'keys_indices' in res:
            keys_indices.append(res['keys_indices'])
            keys_weights.append(res['keys_weights'])
        if 'pred' in res:
            pred = res['pred'].view(-1)
        else:
            pred = res['output'].view(-1)
        preds.append(pred)
        # feas.append(res['fea'])
        if res['att'] is not None:
            if type(res['att']) == np.ndarray:
                pred_att += list(res['att'])
            else:
                pred_att += list(res['att'].detach().cpu().numpy())

        normalized_targets.append(y)
        loss = criterion(pred, y)
        if att_loss:
            p_att = res['att']
            g_att = dt[-2].narrow(-1, 0, p_att.shape[1])
            loss += criterion(p_att, g_att)

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
    if score_range is not None:
        preds = D.recover_scores(normalized_preds, score_range)
        targets = D.recover_scores(normalized_targets, score_range)
        qwk = D.qwk(preds, targets)
    else:
        qwk = -1
    loss = epoch_loss/data_size

    if keys_indices:
        # keys_indices:
        #   [[index_sub1_batch1, index_sub2_batch1],
        #   [index_sub1_batch2, index_sub2_batch2],
        #   ...,
        #   [index_sub1_batchn, index_sub2_batchn]]
        # index_subi_batchj: sahpe(batch_size, )

        # [[index_sub1_batch1, index_sub1_batch2, ...],
        # [index_sub2_batch1, index_sub2_batch2, ...]]
        keys_indices = zip(*keys_indices)

        # [index_sub1, index_sub2, ...]
        keys_indices = [
            torch.cat(ki).detach().cpu().numpy() for ki in keys_indices
        ]
        # [(ind1_1, ind2_1), (ind1_2, ind2_2), ..., (ind1_n, ind2_n)]
        keys_indices = list(zip(*keys_indices))

        keys_weights = zip(*keys_weights)
        keys_weights = [
            torch.cat(kw).detach().cpu().numpy() for kw in keys_weights
        ]
        keys_weights = list(zip(*keys_weights))
    return {
        'qwk': qwk, 'loss': loss,
        'fea': np.zeros([1, 1]),
        'att': pred_att,
        'pred': preds,
        'normalized_pred': normalized_preds,
        'target': targets,
        'normalized_target': normalized_targets,
        'keys_indices': keys_indices,
        'keys_weights': keys_weights
    }
