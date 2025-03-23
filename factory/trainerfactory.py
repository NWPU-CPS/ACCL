class JupyterTrainerFactory(object):
    @classmethod
    def initTrainer(cls, mcfg, dcfg):
        if mcfg.trainer == "triplet":
            from engine.trainer.triplettrainer import JupiterTripletTrainer
            return JupiterTripletTrainer(mcfg, dcfg)
        if mcfg.trainer == "dss":
            from engine.trainer.dsstrainer import JupiterDssTrainer
            return JupiterDssTrainer(mcfg, dcfg)
        raise ValueError("Trainer not registered: {}".format(mcfg.trainer))
