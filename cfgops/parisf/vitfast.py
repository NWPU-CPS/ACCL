import os
from config import mconfig, dconfig


def trainEntry():
    entry = dconfig.DataTripletConfigEntry()
    entry.name = "parisf.triplet"
    entry.queryDir = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/query_images"
    entry.referenceDir = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/reference_images"
    entry.matchFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/gt_matches.csv"
    entry.queryPosFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/query.csv"
    entry.referencePosFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/reference.csv"
    entry.ntriplets = 20000
    return entry


def insampleEvalEntry():
    entry = dconfig.DataTripletConfigEntry()
    entry.name = "parisf.raw"
    entry.queryDir = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/query_images"
    entry.referenceDir = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/reference_images"
    entry.matchFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/gt_matches.csv"
    entry.queryPosFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/query.csv"
    entry.referencePosFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/reference.csv"
    return entry


def evalEntry():
    entry = dconfig.DataTripletConfigEntry()
    entry.name = "parisf.raw"
    entry.queryDir = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Test/query_images"
    entry.referenceDir = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/reference_images"
    entry.matchFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Test/gt_matches.csv"
    entry.queryPosFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Test/query.csv"
    entry.referencePosFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Test/reference.csv"
    return entry


def dcfg(tags):
    dcfg = dconfig.DataConfig()
    dcfg.trainEntry = trainEntry()
    dcfg.insampleEvalEntry = insampleEvalEntry()
    dcfg.evalEntry = evalEntry()
    return dcfg.enrich(tags)


def mcfg(tags):
    mcfg = mconfig.ModelConfig()
    mcfg.root = "/z5s/cv/scan"
    mcfg.modelName = "vit"

    mcfg.phase = "vitsmall16"
    mcfg.device = "cuda:0"
    mcfg.batchSize = 32
    mcfg.outDim = 512
    mcfg.maxEpoch = 50
    mcfg.backboneFreezeEpochs = []
    mcfg.baseLearningRate = 4e-3
    mcfg.evaluator = "momentum"
    mcfg.momentumWindow = 10
    mcfg.maxNorm = 0.1
    mcfg.testSetValidation = True
    mcfg.optimizerType = "SGD"
    mcfg.schedulerType = "COS"

    projectRootDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pretrainedFile = os.path.join(projectRootDir, "resources/pretrained/backbone", "backbone_{}.pth".format(mcfg.phase))
    mcfg.pretrainedBackboneUrl = "file://{}".format(pretrainedFile)

    return mcfg.enrich(tags)
