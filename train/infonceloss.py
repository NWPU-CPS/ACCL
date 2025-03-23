import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn


class InfoNCE(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, qout, rout, dtensor, logit_scale):
        qout = F.normalize(qout, dim=-1)
        rout = F.normalize(rout, dim=-1)

        logits_per_image1 = logit_scale * qout @ rout.T
        logits_per_image2 = logits_per_image1.T
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.mcfg.device)

        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2
        return loss


class InfoNCEWeighted(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.classWeightMin, self.classWeightMax = self.mcfg.classWeightRange
        self.classWeightSpan = self.classWeightMax - self.classWeightMin

    def forward(self, qout, rout, dtensor, logit_scale):
        qout = F.normalize(qout, dim=-1)
        rout = F.normalize(rout, dim=-1)

        logits_per_image1 = logit_scale * qout @ rout.T
        logits_per_image2 = logits_per_image1.T
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.mcfg.device)

        dspan = dtensor.max() - dtensor.min()
        classWeights = (dtensor.max() - dtensor) / dspan * self.classWeightSpan + self.classWeightMin if dspan > 0 else torch.ones(qout.shape[0])
        loss_function = torch.nn.CrossEntropyLoss(weight=classWeights, label_smoothing=0.1).to(self.mcfg.device)

        loss = (loss_function(logits_per_image1, labels) + loss_function(logits_per_image2, labels)) / 2
        return loss


class InfoNCEWeightedDirect(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.classWeightMin, self.classWeightMax = self.mcfg.classWeightRange
        self.classWeightSpan = self.classWeightMax - self.classWeightMin

    def forward(self, qout, rout, dtensor, logit_scale):
        qout = F.normalize(qout, dim=-1)
        rout = F.normalize(rout, dim=-1)

        logits_per_image1 = logit_scale * qout @ rout.T
        logits_per_image2 = logits_per_image1.T
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.mcfg.device)

        dspan = dtensor.max() - dtensor.min()
        classWeights = (dtensor - dtensor.min()) / dspan * self.classWeightSpan + self.classWeightMin if dspan > 0 else torch.ones(qout.shape[0])
        loss_function = torch.nn.CrossEntropyLoss(weight=classWeights, label_smoothing=0.1).to(self.mcfg.device)

        loss = (loss_function(logits_per_image1, labels) + loss_function(logits_per_image2, labels)) / 2
        return loss


class InfoNCEWeightedPlus(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.classWeightMin, self.classWeightMax = self.mcfg.classWeightRange
        self.classWeightSpan = self.classWeightMax - self.classWeightMin
        self.tripletLoss = nn.TripletMarginLoss(margin=self.mcfg.tripletMargin, p=2)

    def forward(self, qout, rout, dtensor, logit_scale):
        qout = F.normalize(qout, dim=-1)
        rout = F.normalize(rout, dim=-1)

        simMat = qout @ rout.T
        logits_per_image1 = logit_scale * simMat
        logits_per_image2 = logits_per_image1.T
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.mcfg.device)

        dspan = dtensor.max() - dtensor.min()
        classWeights = (dtensor.max() - dtensor) / dspan * self.classWeightSpan + self.classWeightMin if dspan > 0 else torch.ones(qout.shape[0])
        loss_function = torch.nn.CrossEntropyLoss(weight=classWeights, label_smoothing=0.1).to(self.mcfg.device)

        infoNceLoss = (loss_function(logits_per_image1, labels) + loss_function(logits_per_image2, labels)) / 2

        # triplet loss with a hard negative
        hardNegativeOutList = []
        _, indexMat = torch.topk(simMat, 2, dim=1)
        for qIndex in range(indexMat.shape[0]):
            for refIndex in indexMat[qIndex]:
                if refIndex == qIndex:
                    continue
                subrout = rout[refIndex, :]
                hardNegativeOutList.append(subrout)
                break

        hardNegativeOut = torch.stack(hardNegativeOutList).to(self.mcfg.device)
        tripletLoss = self.tripletLoss(qout, rout, hardNegativeOut)

        return infoNceLoss + tripletLoss
