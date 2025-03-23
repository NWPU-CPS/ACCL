from config import mconfig, dconfig


def trainEntry():
    entry = dconfig.DataDssConfigEntry()
    entry.name = "parisf.dss"
    entry.queryDir = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/query_images"
    entry.referenceDir = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/reference_images"
    entry.matchFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/gt_matches.csv"
    entry.queryPosFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/query.csv"
    entry.referencePosFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/reference.csv"
    return entry


def insampleEvalEntry():
    entry = dconfig.DataDssConfigEntry()
    entry.name = "parisf.raw"
    entry.queryDir = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/query_images"
    entry.referenceDir = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/reference_images"
    entry.matchFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/gt_matches.csv"
    entry.queryPosFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/query.csv"
    entry.referencePosFile = "/z5s/cps/Paris_fly/Paris_ArcDeTriomphe/Train/reference.csv"
    return entry


def evalEntry():
    entry = dconfig.DataDssConfigEntry()
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
