class JupyterEvaluatorFactory(object):
    @classmethod
    def initEvaluator(cls, mcfg):
        if mcfg.evaluator == "base":
            from eval.precision import JupiterPrecisionEvaluator
            return JupiterPrecisionEvaluator(mcfg)
        if mcfg.evaluator == "momentum":
            from eval.mprecision import JupiterMomentumPrecisionEvaluator
            return JupiterMomentumPrecisionEvaluator(mcfg)
        raise ValueError("Evaluator not registered: {}".format(mcfg))
