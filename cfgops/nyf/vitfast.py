import os
from config import mconfig, dconfig


def trainEntry():
    entry = dconfig.DataTripletConfigEntry()
    entry.name = "nyf.triplet"
    entry.queryDir = "/z5s/cv/cvdata/NewYorkFly/Train/query_images"
    entry.referenceDir = "/z5s/cv/cvdata/NewYorkFly/Train/reference_images"
    entry.matchFile = "/z5s/cv/cvdata/NewYorkFly/Train/gt_match.csv"
    entry.queryPosFile = "/z5s/cv/cvdata/NewYorkFly/Train/query.csv"
    entry.referencePosFile = "/z5s/cv/cvdata/NewYorkFly/Train/reference.csv"
    entry.ntriplets = 20000
    return entry


def insampleEvalEntry():
    entry = dconfig.DataTripletConfigEntry()
    entry.name = "nyf.raw"
    entry.queryDir = "/z5s/cv/cvdata/NewYorkFly/Train/query_images"
    entry.referenceDir = "/z5s/cv/cvdata/NewYorkFly/Train/reference_images"
    entry.matchFile = "/z5s/cv/cvdata/NewYorkFly/Train/gt_match.csv"
    entry.queryPosFile = "/z5s/cv/cvdata/NewYorkFly/Train/query.csv"
    entry.referencePosFile = "/z5s/cv/cvdata/NewYorkFly/Train/reference.csv"
    entry.ntriplets = None
    return entry


def validateEntry():
    entry = dconfig.DataTripletConfigEntry()
    entry.name = "nyf.raw"
    entry.queryDir = "/z5s/cv/cvdata/NewYorkFly/Val/query_images"
    entry.referenceDir = "/z5s/cv/cvdata/NewYorkFly/Val/reference_images"
    entry.matchFile = "/z5s/cv/cvdata/NewYorkFly/Val/gt_match.csv"
    entry.queryPosFile = "/z5s/cv/cvdata/NewYorkFly/Val/query.csv"
    entry.referencePosFile = "/z5s/cv/cvdata/NewYorkFly/Val/reference.csv"
    entry.ntriplets = None
    return entry


def evalEntry():
    entry = dconfig.DataTripletConfigEntry()
    entry.name = "nyf.raw"
    entry.queryDir = "/z5s/cv/cvdata/NewYorkFly/Test/query_images"
    entry.referenceDir = "/z5s/cv/cvdata/NewYorkFly/Test/reference_images"
    entry.matchFile = "/z5s/cv/cvdata/NewYorkFly/Test/gt_match.csv"
    entry.queryPosFile = "/z5s/cv/cvdata/NewYorkFly/Test/query.csv"
    entry.referencePosFile = "/z5s/cv/cvdata/NewYorkFly/Test/reference.csv"
    entry.ntriplets = None
    return entry


def dcfg(tags):
    dcfg = dconfig.DataConfig()
    dcfg.trainEntry = trainEntry()
    dcfg.insampleEvalEntry = insampleEvalEntry()
    dcfg.evalEntry = evalEntry()
    dcfg.dcore = 20
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
