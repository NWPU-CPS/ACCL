import importlib
from misc.misc import nameSplit


class JupiterConfigFactory(object):
    @staticmethod
    def loadConfigModule(cfgname):
        baseName, tags = nameSplit(cfgname, offset=2)
        fullname = "cfgops." + baseName
        module = importlib.import_module(fullname)
        if module is None:
            raise ValueError("Failed to find dcfg: {}".format(fullname))
        return module, tags

    @staticmethod
    def enrichConfig(mcfg, dcfg):
        dcfg.imageSize = mcfg.inputShape
        dcfg.batchSize = mcfg.batchSize
        dcfg.seed = mcfg.seed
        dcfg.device = mcfg.device
        dcfg.maxEpoch = mcfg.maxEpoch
        dcfg.finalize()
        mcfg.dcfg = dcfg
        mcfg.finalize()
        return mcfg, dcfg

    @staticmethod
    def loadConfig(cfgname, mode, nobuf):
        module, tags = JupiterConfigFactory.loadConfigModule(cfgname)
        mcfg = module.mcfg(tags)
        dcfg = module.dcfg(tags)
        mcfg.mode = mode
        mcfg.cfgname = cfgname
        mcfg.nobuf = nobuf
        return JupiterConfigFactory.enrichConfig(mcfg, dcfg)
