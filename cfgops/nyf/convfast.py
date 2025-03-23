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
    return entry


def validateEntry():
    entry = dconfig.DataTripletConfigEntry()
    entry.name = "nyf.raw"
    entry.queryDir = "/z5s/cv/cvdata/NewYorkFly/Val/query_images"
    entry.referenceDir = "/z5s/cv/cvdata/NewYorkFly/Val/reference_images"
    entry.matchFile = "/z5s/cv/cvdata/NewYorkFly/Val/gt_match.csv"
    entry.queryPosFile = "/z5s/cv/cvdata/NewYorkFly/Val/query.csv"
    entry.referencePosFile = "/z5s/cv/cvdata/NewYorkFly/Val/reference.csv"
    return entry


def evalEntry():
    entry = dconfig.DataTripletConfigEntry()
    entry.name = "nyf.raw"
    entry.queryDir = "/z5s/cv/cvdata/NewYorkFly/Test/query_images"
    entry.referenceDir = "/z5s/cv/cvdata/NewYorkFly/Test/reference_images"
    entry.matchFile = "/z5s/cv/cvdata/NewYorkFly/Test/gt_match.csv"
    entry.queryPosFile = "/z5s/cv/cvdata/NewYorkFly/Test/query.csv"
    entry.referencePosFile = "/z5s/cv/cvdata/NewYorkFly/Test/reference.csv"
    return entry


def dcfg(tags):
    dcfg = dconfig.DataConfig()
    dcfg.trainEntry = trainEntry()
    dcfg.validateEntry = validateEntry()
    dcfg.insampleEvalEntry = insampleEvalEntry()
    dcfg.evalEntry = evalEntry()
    return dcfg


def mcfg(tags):
    mcfg = mconfig.ModelConfig()
    mcfg.root = "/z5s/cv/scan"
    mcfg.trainer = "triplet"
    mcfg.modelName = "timm"
    mcfg.phase = "convnext_base.fb_in22k_ft_in1k_384"
    mcfg.inputShape = (383, 383)

    mcfg.device = "cuda:0"
    mcfg.batchSize = 32
    mcfg.maxEpoch = 50
    mcfg.backboneFreezeEpochs = []
    mcfg.baseLearningRate = 4e-3
    mcfg.testSetValidation = True

    for tag in tags:
        tokens = tag.split("@")
        match tokens[0]:
            case "cuda":
                mcfg.device = "cuda:{}".format(tokens[1])
            case "batch":
                mcfg.batchSize = int(tokens[1])
            case "lr":
                mcfg.baseLearningRate = float(tokens[1]) * 1e-3

    return mcfg
