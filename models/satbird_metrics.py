# Src code for all metrics used
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import Metric


eps = 1e-7


class CustomCrossEntropyLoss:
    def __init__(self, lambd_pres=1, lambd_abs=1):
        super().__init__()
        # print('in my custom')
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres

    def __call__(self, pred, target, reduction='mean'):
        """
        target: ground truth
        pred: prediction
        reduction: mean, sum, none
        """
        loss = (-self.lambd_pres * target * torch.log(pred + eps) - self.lambd_abs * (1 - target) * torch.log(
            1 - pred + eps))
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:  # reduction = None
            loss = loss

        return loss


class RMSLELoss(nn.Module):
    """
    root mean squared log error
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(target + 1)))


class CustomFocalLoss:
    def __init__(self, alpha=1, gamma=2):
        """
        build on top of binary cross entropy as implemented before
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, pred, target):
        ce_loss = (- target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps)).mean()
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class CustomCrossEntropy(Metric):
    def __init__(self, lambd_pres=1, lambd_abs=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres
        self.add_state("correct", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        target: target distribution
        pred: predicted distribution
        """
        self.correct += (-self.lambd_pres * target * torch.log(pred) - self.lambd_abs * (1 - target) * torch.log(
            1 - pred)).sum()
        self.total += target.numel()

    def compute(self):
        return (self.correct / self.total)


class WeightedCustomCrossEntropyLoss:
    def __init__(self, lambd_pres=1, lambd_abs=1):
        super().__init__()
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres

    def __call__(self, pred, target, weights=1):
        loss = (weights * (
                -self.lambd_pres * target * torch.log(pred + eps) - self.lambd_abs * (1 - target) * torch.log(
            1 - pred + eps))).mean()

        return loss


class CustomKL(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")

    def update(self, p: torch.Tensor, q: torch.Tensor):
        """
        p: target distribution
        q: predicted distribution
        """
        self.correct += (torch.nansum(p * torch.log(p / q)) + torch.nansum((1 - p) * torch.log((1 - p) / (1 - q))))
        self.total += p.numel()

    def compute(self):
        return (self.correct / self.total)


class Presence_k(nn.Module):
    """
    compare accuracy by binarizing targets  1 if species are present with proba > k
    """

    def __init__(self, k):
        super().__init__()
        self.k = k

    def __call__(self, target, pred):
        pres = ((pred > self.k) == (target > self.k)).mean()
        return (pres)


class CustomTopK(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, preds: torch.Tensor):

        assert preds.shape == target.shape
        non_zero_counts = torch.count_nonzero(target, dim=1)
        for i, elem in enumerate(target):
            ki = non_zero_counts[i]
            v_pred, i_pred = torch.topk(preds[i], k=ki)
            v_targ, i_targ = torch.topk(elem, k=ki)
            if ki == 0:
                pass
            else:
                count = torch.tensor(len([k for k in i_pred if k in i_targ]))
                self.correct += count / ki
                self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class CustomTop10(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, preds: torch.Tensor):

        assert preds.shape == target.shape
        non_zero_counts = torch.count_nonzero(target, dim=1)
        for i, elem in enumerate(target):
            ki = non_zero_counts[i]
            if ki >= 10:
                v_pred, i_pred = torch.topk(preds[i], k=10)
                v_targ, i_targ = torch.topk(elem, k=10)
            else:
                v_pred, i_pred = torch.topk(preds[i], 10)
                v_targ, i_targ = torch.topk(elem, ki)
            if ki == 0:
                pass
            else:
                count = torch.tensor(len([k for k in i_pred if k in i_targ]))
                if ki >= 10:
                    self.correct += count / 10
                else:
                    self.correct += count / ki
                self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class CustomTop30(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, preds: torch.Tensor):

        assert preds.shape == target.shape
        non_zero_counts = torch.count_nonzero(target, dim=1)
        for i, elem in enumerate(target):
            ki = non_zero_counts[i]
            if ki >= 30:
                v_pred, i_pred = torch.topk(preds[i], k=30)
                v_targ, i_targ = torch.topk(elem, k=30)
            else:
                v_pred, i_pred = torch.topk(preds[i], 30)
                v_targ, i_targ = torch.topk(elem, k=ki)
            if ki == 0:
                pass
            else:
                count = torch.tensor(len([k for k in i_pred if k in i_targ]))
                if ki >= 30:
                    self.correct += count / 30
                else:
                    self.correct += count / ki
                self.total += 1

    def compute(self):
        return self.correct.float() / self.total


def get_metric(metric):
    """Returns the transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if metric.name == "mae" and not metric.ignore is True:
        return torchmetrics.MeanAbsoluteError()
    elif metric.name == "mse" and not metric.ignore is True:
        return torchmetrics.MeanSquaredError()
    elif metric.name == "topk" and not metric.ignore is True:
        return CustomTopK()
    elif metric.name == "top10" and not metric.ignore is True:
        return CustomTop10()
    elif metric.name == "top30" and not metric.ignore is True:
        return CustomTop30()
    elif metric.name == "ce" and not metric.ignore is True:
        return CustomCrossEntropy(metric.lambd_pres, metric.lambd_abs)
    elif metric.name == 'r2' and not metric.ignore is True:
        return torchmetrics.ExplainedVariance(
            multioutput='variance_weighted')
    elif metric.name == "kl" and not metric.ignore is True:
        return CustomKL()
    elif metric.name == "accuracy" and not metric.ignore is True:
        return torchmetrics.Accuracy()
    elif metric.ignore is True:
        return None
    else:
        return (None)  # raise ValueError("Unknown metric_item {}".format(metric))


def get_metrics(config):
    metrics = []
    for m in config.losses.metrics:
        metrics.append((m.name, get_metric(m), m.scale))
    metrics = [(a, b, c) for (a, b, c) in metrics if b is not None]
    return metrics
