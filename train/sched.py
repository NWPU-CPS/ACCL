import torch
import math


class CosineScheduler(object):
    def __init__(self, mcfg, opt):
        self.mcfg = mcfg
        self.opt = opt
        self.maxEpoch = self.mcfg.maxEpoch
        self.baseLearningRate = self.mcfg.baseLearningRate
        self.minLearningRate = self.mcfg.minLearningRate

        self.warmupEpochRatio = 0.05
        self.warmupLearningRateRatio = 0.1
        self.noAugEpochRatio = 0.05
        self.stepNum = 10

        self.warmupTotalEpochs = min(max(self.warmupEpochRatio * self.maxEpoch, 1), 3)
        self.warmupLearningRate = max(self.baseLearningRate * self.warmupLearningRateRatio, 1e-6)
        self.noAugEpochs = min(max(self.noAugEpochRatio * self.maxEpoch, 1), 15)

    def getLearningRate(self, epoch):
        if epoch <= self.warmupTotalEpochs:
            return (self.baseLearningRate - self.warmupLearningRate) * pow(epoch / float(self.warmupTotalEpochs), 2) + self.warmupLearningRate
        elif epoch >= self.maxEpoch - self.noAugEpochs:
            return self.minLearningRate
        else:
            return self.minLearningRate + 0.5 * (self.baseLearningRate - self.minLearningRate) * (
                1.0 + math.cos(
                    math.pi
                    * (epoch - self.warmupTotalEpochs)
                    / (self.maxEpoch - self.warmupTotalEpochs - self.noAugEpochs)
                )
            )

    def onEpochUpdate(self, epoch):
        learningRate = self.getLearningRate(epoch)
        for param_group in self.opt.param_groups:
            param_group["lr"] = learningRate

    def step(self):
        return


class JupiterLearningRateSchedulerFactory(object):
    @staticmethod
    def initScheduler(mcfg, opt, steps):
        match mcfg.schedulerType:
            case "COS":
                return CosineScheduler(mcfg, opt)
            case other:
                raise ValueError("Invalid optimizer type: {}".format(mcfg.optimizerType))
