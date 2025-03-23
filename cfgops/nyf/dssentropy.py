from config import mconfig, dconfig


def trainEntry():
    entry = dconfig.DataDssConfigEntry()
    entry.name = "nyf.dssentropy"
    entry.queryDir = "/z5s/cv/cvdata/NewYorkFly/Train/query_images"
    entry.referenceDir = "/z5s/cv/cvdata/NewYorkFly/Train/reference_images"
    entry.matchFile = "/z5s/cv/cvdata/NewYorkFly/Train/gt_match.csv"
    entry.queryPosFile = "/z5s/cv/cvdata/NewYorkFly/Train/query.csv"
    entry.referencePosFile = "/z5s/cv/cvdata/NewYorkFly/Train/reference.csv"
    return entry


def insampleEvalEntry():
    entry = dconfig.DataDssConfigEntry()
    entry.name = "nyf.raw"
    entry.queryDir = "/z5s/cv/cvdata/NewYorkFly/Train/query_images"
    entry.referenceDir = "/z5s/cv/cvdata/NewYorkFly/Train/reference_images"
    entry.matchFile = "/z5s/cv/cvdata/NewYorkFly/Train/gt_match.csv"
    entry.queryPosFile = "/z5s/cv/cvdata/NewYorkFly/Train/query.csv"
    entry.referencePosFile = "/z5s/cv/cvdata/NewYorkFly/Train/reference.csv"
    return entry


def validateEntry():
    entry = dconfig.DataDssConfigEntry()
    entry.name = "nyf.raw"
    entry.queryDir = "/z5s/cv/cvdata/NewYorkFly/Val/query_images"
    entry.referenceDir = "/z5s/cv/cvdata/NewYorkFly/Val/reference_images"
    entry.matchFile = "/z5s/cv/cvdata/NewYorkFly/Val/gt_match.csv"
    entry.queryPosFile = "/z5s/cv/cvdata/NewYorkFly/Val/query.csv"
    entry.referencePosFile = "/z5s/cv/cvdata/NewYorkFly/Val/reference.csv"
    return entry


def evalEntry():
    entry = dconfig.DataDssConfigEntry()
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
    return dcfg.enrich(tags)


def mcfg(tags):
    mcfg = mconfig.ModelConfig()
    mcfg.root = "/z5s/cv/scan"
    mcfg.trainer = "dss"
    mcfg.modelName = "timm"
    mcfg.phase = "convnext_base.fb_in22k_ft_in1k_384"
    mcfg.inputShape = (383, 383)
    mcfg.lossName = "infoNCEWeighted"
    mcfg.evaluator = "momentum"

    mcfg.device = "cuda:0"
    mcfg.batchSize = 32
    mcfg.maxEpoch = 50
    mcfg.backboneFreezeEpochs = []
    mcfg.baseLearningRate = 4e-3
    mcfg.testSetValidation = True

    return mcfg.enrich(tags)
