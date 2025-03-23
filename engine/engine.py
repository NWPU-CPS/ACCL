from misc.log import log
from misc.misc import setSeedGlobal
from engine.evaluator import JupiterEvaluator
from factory.configfactory import JupiterConfigFactory
from factory.trainerfactory import JupyterTrainerFactory


class JupiterEngine(object):
    def __init__(self, mode, cfgname, nobuf):
        self.mode = mode
        self.cfgname = cfgname
        self.mcfg, self.dcfg = JupiterConfigFactory.loadConfig(cfgname, mode, nobuf)

    def initialize(self):
        import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
        log.inf("Jupiter engine initializing with cfgname <{}>...".format(self.cfgname))
        setSeedGlobal(self.mcfg.seed)

    def run(self):
        self.initialize()
        if self.mode in ("train", "pipe"):
            self.runTraining()
        if self.mode in ("eval", "pipe"):
            self.runEvaluation()

    def runTraining(self):
        trainer = JupyterTrainerFactory.initTrainer(self.mcfg, self.dcfg)
        trainer.run()

    def runEvaluation(self):
        evaluator = JupiterEvaluator(self.mcfg, self.dcfg)
        evalDf, smrDf, detailedDf = evaluator.run(save=True)
        self.view(evalDf, smrDf, detailedDf)

    def view(self, evalDf, smrDf, detailedDf):
        import pdb; pdb.set_trace()
