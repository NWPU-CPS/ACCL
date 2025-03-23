import os
from misc.log import log
from factory.evalfactory import JupyterEvaluatorFactory
from factory.dlfactory import JupyterDataLoaderFactory


class JupiterEvaluator(object):
    def __init__(self, mcfg, dcfg):
        self.mcfg = mcfg
        self.dcfg = dcfg
        self.evalDir = self.mcfg.evalDir()
        self.precisionEvaluator = JupyterEvaluatorFactory.initEvaluator(mcfg)

    def initDataLoader(self):
        return JupyterDataLoaderFactory.getEvalDataLoader(self.dcfg, self.dcfg.evalEntry)

    def initModel(self):
        from factory.modelfactory import JupyterModelFactory
        modelFile = self.mcfg.modelLoadPath()
        return JupyterModelFactory.createEvalModel(mcfg=self.mcfg, modelFile=modelFile).to(self.mcfg.device)

    def run(self, save=False):
        log.cyan("Jupiter evaluator running...")

        dataLoader = self.initDataLoader()
        model = self.initModel()
        evalData = dataLoader.load()
        evalDf, smrDf, detailedDf = self.precisionEvaluator.eval(model, evalData)

        if save:
            evalFile = os.path.join(self.evalDir, "eval.csv")
            smrFile = os.path.join(self.evalDir, "smr.csv")
            detailFile = os.path.join(self.evalDir, "detail.csv")
            evalDf.to_csv(evalFile, index=False, na_rep="nan")
            smrDf.to_csv(smrFile, index=False, na_rep="nan")
            detailedDf.to_csv(detailFile, index=False, na_rep="nan")

        return evalDf, smrDf, detailedDf
