import torch
import torch.nn.functional as F


class LabelSmoothing(torch.nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target, statistics=False):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if statistics:
            return loss.mean(), {'loss_ce': nll_loss.detach(), 'loss_uniform': smooth_loss.detach()}
        else:
            return loss.mean()


class SampleLabelSmoothing(LabelSmoothing):
    """
    Sample-wise NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__(smoothing)

    def forward(self, x, target, prob=None, statistics=False):
        smoothing = self.smoothing
        self.smoothing = (1 - prob) * self.smoothing
        if statistics:
            loss, stats = super().forward(x, target, True)
            stats['smooth'] = self.smoothing
            self.smoothing = smoothing
            return loss, stats
        else:
            loss = super().forward(x, target, False)
            self.smoothing = smoothing
            return loss
