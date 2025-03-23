import os
import torch


class ModelConfig(object):
    def __init__(self):
        self.user = None
        self.seed = 859
        self.device = None

        # model setup
        self.modelName = "timm"
        self.phase = "convnext_base.fb_in22k_ft_in1k_384"
        self.inputShape = (383, 383)
        self.reshuffleEpochs = 4

        self.SPT = False
        self.LSA = False
        self.outDim = 256
        self.pretrainedBackboneUrl = None

        # train setup
        self.lossName = "infoNCEWeighted"
        self.classWeightRange = [0.3, 1.5]
        self.tripletWeight = 0.1

        self.trainer = "dss"
        self.testSetValidation = False

        self.tripletMargin = 0.3
        self.hardnetMargin = 1.5

        self.optimizerType = "SGD"
        self.optimizerMomentum = 0.937
        self.optimizerWeightDecay = 5e-4
        self.maxNorm = 0.1

        self.schedulerType = "COS"
        self.startEpoch = 0
        self.warmupEpochs = 10
        self.baseLearningRate = 1e-4
        self.minLearningRate = self.baseLearningRate * 1e-2

        self.maxEpoch = 50
        self.backboneFreezeEpochs = []
        self.numBlockFreezed = 10
        self.batchSize = 64

        # eval setup
        self.evaluator = "base"
        self.kNeighbors = 5
        self.momentumWindow = 20

        # enriched by factory
        self.mode = None
        self.root = None
        self.cfgname = None
        self.nobuf = False
        self.dcfg = None

    def enrich(self, tags):
        for tag in tags:
            tokens = tag.split("@")
            match tokens[0]:
                case "cuda":
                    self.device = "cuda:{}".format(tokens[1])
                case "phase":
                    self.phase = tokens[1]
                case "batch":
                    self.batchSize = int(tokens[1])
                case "outDim":
                    self.outDim = int(tokens[1])
                case "lr":
                    self.baseLearningRate = float(tokens[1]) * 1e-4
                case "reshuffle":
                    self.reshuffleEpochs = int(tokens[1])
                case "maxnorm":
                    self.maxNorm = float(tokens[1])
                case "tmargin":
                    self.tripletMargin = float(tokens[1]) * 0.1
                case "hmargin":
                    self.hardnetMargin = float(tokens[1]) * 0.1
                case "tweight":
                    self.tripletWeight = float(tokens[1]) * 0.1
                case "maxepoch":
                    self.maxEpoch = int(tokens[1])
                case "loss":
                    self.lossName = tokens[1]
        return self

    def finalize(self):
        self.user = os.getenv("USER")
        if self.user is None or len(self.user) == 0:
            raise ValueError("User not found")
        if self.root is None:
            raise ValueError("Root not set")
        if self.mode is None:
            raise ValueError("Mode not set")
        if self.cfgname is None:
            raise ValueError("Cfgname not set")
        if self.phase is None:
            raise ValueError("Phase not set")
        if self.device is None:
            self.device = torch.device("cuda" if self.cuda else "cpu")

        self.cacheDir()
        self.evalDir()
        return self

    def backboneUrl(self):
        return self.pretrainedBackboneUrl

    def cacheDir(self):
        dirpath = os.path.join(self.root, self.user, self.cfgname, "__cache__")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath

    def evalDir(self):
        dirpath = os.path.join(self.root, self.user, self.cfgname, "eval")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath

    def modelLoadPath(self):
        if os.path.exists(self.epochBestWeightsPath()):
            return self.epochBestWeightsPath()
        return self.epochCachePath()

    def onnxSavePath(self):
        return os.path.join(self.cacheDir(), "lastest.onnx")

    def epochCachePath(self):
        return os.path.join(self.cacheDir(), "last_epoch_weights.pth")

    def epochInfoPath(self):
        return os.path.join(self.cacheDir(), "info.txt")

    def epochBestWeightsPath(self):
        return os.path.join(self.cacheDir(), "best_weights.pth")
