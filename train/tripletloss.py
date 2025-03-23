import torch
import torch.nn as nn
from misc.log import log


class TripletLoss(nn.Module):
    def __init__(self, margin=None):
        super().__init__()
        self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.Loss(anchor, positive, negative)


class CorrelationPenaltyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        mean1 = torch.mean(input, dim=0)
        zeroed = input - mean1.expand_as(input)
        cor_mat = torch.bmm(torch.t(zeroed).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
        d = torch.diag(torch.diag(cor_mat))
        no_diag = cor_mat - d
        d_sq = no_diag * no_diag
        return torch.sqrt(d_sq.sum()) / input.size(0)


class HardNetLoss(nn.Module):
    def __init__(self, anchor_swap=False, anchor_ave=False, margin=1.0, batch_reduce="min", loss_type="triplet_margin"):
        super().__init__()
        self.anchor_swap = anchor_swap
        self.anchor_ave = anchor_ave
        self.margin = margin
        self.batch_reduce = batch_reduce
        self.loss_type = loss_type

    def distance_matrix_vector(self, anchor, positive):
        """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""
        return torch.cdist(anchor, positive)

    def forward(self, anchor, positive):
        eps = 1e-8
        dist_matrix = self.distance_matrix_vector(anchor, positive) + eps
        eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).to(anchor.device)

        pos1 = torch.diag(dist_matrix)
        dist_without_min_on_diag = dist_matrix + eye * 10
        mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * -1
        mask = mask.type_as(dist_without_min_on_diag)*10
        dist_without_min_on_diag = dist_without_min_on_diag + mask

        if self.batch_reduce == 'min':
            min_neg = torch.min(dist_without_min_on_diag, 1)[0]
            if self.anchor_swap:
                min_neg2 = torch.min(dist_without_min_on_diag, 0)[0]
                min_neg = torch.min(min_neg, min_neg2)
            min_neg = min_neg
            pos = pos1
        elif self.batch_reduce == 'average':
            pos = pos1.repeat(anchor.size(0)).view(-1, 1).squeeze(0)
            min_neg = dist_without_min_on_diag.view(-1, 1)
            if self.anchor_swap:
                min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1, 1)
                min_neg = torch.min(min_neg,min_neg2)
            min_neg = min_neg.squeeze(0)
        elif self.batch_reduce == 'random':
            idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).to(anchor.device())
            min_neg = dist_without_min_on_diag.gather(1, idxs.view(-1, 1))
            if self.anchor_swap:
                min_neg2 = torch.t(dist_without_min_on_diag).gather(1, idxs.view(-1, 1))
                min_neg = torch.min(min_neg,min_neg2)
            min_neg = torch.t(min_neg).squeeze(0)
            pos = pos1
        else:
            raise ValueError("Unknown batch reduce mode: {}".format(self.batch_reduce))

        if self.loss_type == "triplet_margin":
            loss = torch.clamp(self.margin + pos - min_neg, min=0.0)
            # criterion = TripletLoss(margin=0.3)
            # loss = torch.clamp(criterion(anchor[0], pos, min_neg), min=0.0)
        elif self.loss_type == 'softmax':
            exp_pos = torch.exp(2.0 - pos)
            exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps
            loss = - torch.log( exp_pos / exp_den )
        elif self.loss_type == 'contrastive':
            loss = torch.clamp(self.margin - min_neg, min=0.0) + pos
        else:
            raise ValueError("Unknown loss type: {}".format(self.loss_type))

        loss = torch.mean(loss)
        if torch.isnan(loss):
            log.red("loss is nan")
            import pdb; pdb.set_trace()
        return loss


class SyntheticLoss(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.tripletLoss = TripletLoss(margin=0.3)
        self.corrPenaltyLoss = CorrelationPenaltyLoss()
        self.hardLoss = HardNetLoss(
            margin=1.5,
            anchor_swap=True,
            anchor_ave=False,
            batch_reduce="min",
            loss_type="triplet_margin",
        )
        self.epoch = 0

    def onEpochUpdate(self, epoch):
        self.epoch = epoch
        self.corrPenaltyLossWeight = (self.mcfg.maxEpoch - epoch) / self.mcfg.maxEpoch * (1 - 0.1) + 0.1

    def forward(self, anchor, positive, negative):
        tripletLoss = self.tripletLoss(anchor, positive, negative)
        corrPenaltyLoss = (self.corrPenaltyLoss(anchor) + self.corrPenaltyLoss(positive) + self.corrPenaltyLoss(negative)) * self.corrPenaltyLossWeight
        hardLoss = self.hardLoss(anchor, positive)
        return tripletLoss + corrPenaltyLoss + hardLoss
