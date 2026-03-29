import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score
import torch
import os
from sklearn.preprocessing import label_binarize


def seed_torch(seed=2021):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def ema_update(model, targ_model, mm=0.9999):
    assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm

    for param_q, param_k in zip(model.parameters(), targ_model.parameters()):
        param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)


def patch_shuffle(x, group=0, g_idx=None, return_g_idx=False):
    b, p, n = x.size()
    ps = torch.tensor(list(range(p)))

    if group > H or group <= 0:
        return group_shuffle(x, group)

    H, W = int(np.ceil(np.sqrt(p))), int(np.ceil(np.sqrt(p)))
    _n = -H % group
    H, W = H + _n, W + _n
    add_length = H * W - p
    ps = torch.cat([ps, torch.tensor([-1 for _ in range(add_length)])])

    ps = ps.reshape((group, H // group, group, W // group))
    ps = torch.einsum('hpwq->hwpq', ps)
    ps = ps.reshape((group ** 2, H // group, W // group))

    if g_idx is None:
        g_idx = torch.randperm(ps.size(0))
    ps = ps[g_idx]

    ps = ps.reshape((group, group, H // group, W // group))
    ps = torch.einsum('hwpq->hpwq', ps)
    ps = ps.reshape((H, W))
    idx = ps[ps >= 0].view(p)

    if return_g_idx:
        return x[:, idx.long()], g_idx
    else:
        return x[:, idx.long()]


def group_shuffle(x, group=0):
    b, p, n = x.size()
    ps = torch.tensor(list(range(p)))

    if group > 0 and group < p:
        _pad = -p % group
        ps = torch.cat([ps, torch.tensor([-1 for _ in range(_pad)])])
        ps = ps.view(group, -1)
        g_idx = torch.randperm(ps.size(0))
        ps = ps[g_idx]
        idx = ps[ps >= 0].view(p)
    else:
        idx = torch.randperm(p)

    return x[:, idx.long()]


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    labels = np.array(dataset.slide_label)
    label_uni = set(dataset.slide_label)
    weight_per_class = [N / len(labels[labels == c]) for c in label_uni]
    weight = [0] * int(N)

    for idx in range(len(dataset)):
        y = dataset.slide_label[idx]
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def five_scores(bag_labels, bag_hat, bag_prob):
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import torchmetrics

    threshold_optimal = 0.5

    if isinstance(bag_prob, list):
        bag_prob = np.array(bag_prob)
    if isinstance(bag_hat, list):
        bag_hat = torch.tensor(bag_hat)
    if isinstance(bag_labels, list):
        bag_labels = torch.tensor(bag_labels)

    bag_labels = bag_labels.view(-1).cpu()
    bag_hat = bag_hat.view(-1).cpu()
    y_true = bag_labels.numpy()

    if bag_prob.shape[1] > 1:
        auc_per_class = roc_auc_score(y_true, bag_prob, multi_class='ovr', average=None)
        auc_avg = np.mean(auc_per_class)
    else:
        auc_avg = roc_auc_score(y_true, bag_prob)
        auc_per_class = [auc_avg]

    metric_collection = torchmetrics.MetricCollection([
        torchmetrics.Accuracy(task='multiclass', num_classes=4, average=None),
        torchmetrics.Precision(task='multiclass', num_classes=4, average=None),
        torchmetrics.Recall(task='multiclass', num_classes=4, average=None),
        torchmetrics.F1Score(task='multiclass', num_classes=4, average=None),
        torchmetrics.Specificity(task='multiclass', num_classes=4, average=None)
    ])

    res = metric_collection(bag_hat, bag_labels)

    per_class_metrics = {
        'accuracy': res['MulticlassAccuracy'].numpy(),
        'precision': res['MulticlassPrecision'].numpy(),
        'recall': res['MulticlassRecall'].numpy(),
        'fscore': res['MulticlassF1Score'].numpy(),
        'specificity': res['MulticlassSpecificity'].numpy(),
        'auc': auc_per_class
    }

    metrics = torchmetrics.MetricCollection([
        torchmetrics.Accuracy(task='multiclass', num_classes=4, average='micro'),
        torchmetrics.F1Score(task='multiclass', num_classes=4, average='macro'),
        torchmetrics.Recall(task='multiclass', average='macro', num_classes=4),
        torchmetrics.Precision(task='multiclass', average='macro', num_classes=4),
        torchmetrics.Specificity(task='multiclass', average='macro', num_classes=4),
    ])

    metrics.update(bag_hat, bag_labels)
    results = metrics.compute()
    metrics.reset()

    avg_metrics = (
        results['MulticlassAccuracy'].item(),
        auc_avg,
        results['MulticlassPrecision'].item(),
        results['MulticlassRecall'].item(),
        results['MulticlassF1Score'].item(),
        results['MulticlassSpecificity'].item(),
        threshold_optimal
    )

    return avg_metrics, per_class_metrics


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=50, verbose=False, save_best_model_stage=0.):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_best_model_stage = save_best_model_stage

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        score = -val_loss if epoch >= self.save_best_model_stage else 0.

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def state_dict(self):
        return {
            'patience': self.patience,
            'stop_epoch': self.stop_epoch,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min
        }

    def load_state_dict(self, dict):
        self.patience = dict['patience']
        self.stop_epoch = dict['stop_epoch']
        self.verbose = dict['verbose']
        self.counter = dict['counter']
        self.best_score = dict['best_score']
        self.early_stop = dict['early_stop']
        self.val_loss_min = dict['val_loss_min']

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.val_loss_min = val_loss