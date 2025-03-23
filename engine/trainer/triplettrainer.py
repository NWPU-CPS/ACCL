import os
import torch
from misc.log import log
from tqdm import tqdm
from eval.precision import JupiterPrecisionEvaluator
from factory.dlfactory import JupyterDataLoaderFactory


class JupiterTripletTrainer(object):
    def __init__(self, mcfg, dcfg):
        self.mcfg = mcfg
        self.dcfg = dcfg
        self.epochCacheFile = self.mcfg.epochCachePath()
        self.epochInfoFile = self.mcfg.epochInfoPath()
        self.bestWeightFile = self.mcfg.epochBestWeightsPath()
        self.checkpointFiles = [
            self.epochCacheFile,
            self.epochInfoFile,
        ]
        self.precisionEvaluator = JupiterPrecisionEvaluator(mcfg)
        self.bestScore = 0

    def initTrainDataLoader(self):
        return JupyterDataLoaderFactory.getTrainDataLoader(self.dcfg, self.dcfg.trainEntry)

    def initTrainDataLoaderRaw(self):
        """
        Used to calculate in-sample precision.
        """
        return JupyterDataLoaderFactory.getEvalDataLoader(self.dcfg, self.dcfg.insampleEvalEntry)

    def initValidationDataLoader(self):
        if self.mcfg.testSetValidation:
            log.inf("Test set validation is ON")
            return JupyterDataLoaderFactory.getEvalDataLoader(self.dcfg, self.dcfg.evalEntry)
        else:
            return JupyterDataLoaderFactory.getEvalDataLoader(self.dcfg, self.dcfg.validateEntry)

    def initModel(self):
        from factory.modelfactory import JupyterModelFactory
        model = JupyterModelFactory.createTrainModel(mcfg=self.mcfg)
        startEpoch = 0

        if not self.mcfg.nobuf and all(os.path.exists(x) for x in self.checkpointFiles): # resume from checkpoint to continue training
            model.load(self.epochCacheFile)
            startEpoch = None
            with open(self.epochInfoFile) as f:
                lines = f.readlines()
                for line in lines:
                    tokens = line.split("=")
                    if len(tokens) != 2:
                        continue
                    if tokens[0] == "last_saved_epoch":
                        startEpoch = int(tokens[1])
                    if tokens[0] == "best_score":
                        self.bestScore = float(tokens[1])
            if startEpoch is None:
                raise ValueError("Failed to load last epoch info from file: {}".format(self.epochInfoFile))
            if startEpoch < self.mcfg.maxEpoch:
                log.yellow("Checkpoint loaded: resuming from epoch {}/{}".format(startEpoch, self.mcfg.maxEpoch))

        return model.to(self.mcfg.device), startEpoch

    def initLoss(self):
        from train.tripletloss import SyntheticLoss
        return SyntheticLoss(mcfg=self.mcfg).to(self.mcfg.device)

    def initOptimizer(self, model):
        from train.opt import JupiterOptimizerFactory
        return JupiterOptimizerFactory.initOptimizer(self.mcfg, model)

    def initScheduler(self, opt, steps):
        from train.sched import JupiterLearningRateSchedulerFactory
        return JupiterLearningRateSchedulerFactory.initScheduler(self.mcfg, opt, steps)

    def fitOneEpoch(self, model, loss, dataLoader, optimizer, scheduler, epoch):
        trainLoss = 0
        model.train()
        model.setInferenceMode(False)
        numBatches = len(dataLoader)
        progressBar = tqdm(total=numBatches, desc="Epoch {}/{}".format(epoch + 1, self.mcfg.maxEpoch), postfix=dict, mininterval=0.5, ascii=False, ncols=100)

        for batchIndex, items in enumerate(dataLoader):
            qtensor, rtensor, ntensor = [x.to(self.mcfg.device) for x in items]
            qout = model(qtensor)
            rout = model(rtensor)
            nout = model(ntensor)

            optimizer.zero_grad()
            stepLoss = loss(qout, rout, nout)
            stepLoss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.mcfg.maxNorm)
            optimizer.step()

            trainLoss += stepLoss.item()
            progressBar.set_postfix(trainLoss=trainLoss / (batchIndex + 1), backboneFreezed=model.backboneFreezed, lr=optimizer.param_groups[0]["lr"])
            progressBar.update(1)

        progressBar.close()
        return trainLoss

    def validate(self, model, insampleEvalData, evalData):
        model.eval()
        model.setInferenceMode(True)
        # in-sample on train set
        insampleEvalDf, smrDf, _ = self.precisionEvaluator.eval(model, insampleEvalData)
        insamplePrecision = smrDf.loc[smrDf["order"] == 0, "precision"].iloc[0]
        # out-sample on validation set
        _, smrDf, _ = self.precisionEvaluator.eval(model, evalData)
        evalPrecision = smrDf.loc[smrDf["order"] == 0, "precision"].iloc[0]
        return insamplePrecision, evalPrecision, insampleEvalDf

    def run(self):
        log.cyan("Jupiter trainer running...")

        model, startEpoch = self.initModel()
        if startEpoch >= self.mcfg.maxEpoch:
            log.inf("Training skipped")
            return

        loss = self.initLoss()
        opt = self.initOptimizer(model)
        trainLoader = self.initTrainDataLoader()
        trainRawLoader = self.initTrainDataLoaderRaw()
        valLoader = self.initValidationDataLoader()
        scheduler = self.initScheduler(opt, len(trainLoader))
        insampleEvalData = trainRawLoader.load()
        evalData = valLoader.load()

        insampleScore, valScore, _ = self.validate(
            model=model,
            insampleEvalData=insampleEvalData,
            evalData=evalData,
        )
        log.inf("Init model score: in-{:.4f}/out-{:.4f}".format(insampleScore, valScore))

        for epoch in range(startEpoch, self.mcfg.maxEpoch):
            scheduler.onEpochUpdate(epoch)
            loss.onEpochUpdate(epoch)
            model.setFreezeBackbone(epoch in self.mcfg.backboneFreezeEpochs)
            trainLoss = self.fitOneEpoch(
                model=model,
                loss=loss,
                dataLoader=trainLoader,
                optimizer=opt,
                scheduler=scheduler,
                epoch=epoch,
            )
            insampleScore, valScore, insampleEvalDf = self.validate(
                model=model,
                insampleEvalData=insampleEvalData,
                evalData=evalData,
            )
            self.epochSave(epoch, model, trainLoss, insampleScore, valScore)
            if epoch > 0 and epoch % self.mcfg.reshuffleEpochs == 0:
                trainLoader.dataset.resample(epoch, insampleEvalDf)

        log.inf("Jupiter trainer DONE training at epoch {}".format(self.mcfg.maxEpoch))

    def epochSave(self, epoch, model, trainLoss, insampleScore, valScore):
        model.save(self.epochCacheFile)

        if self.bestScore < valScore:
            model.save(self.bestWeightFile)
            self.bestScore = valScore
            log.green("Best weights updated at epoch {} with score {:.4f}".format(epoch + 1, valScore))

        with open(self.epochInfoFile, "w") as f:
            f.write("last_saved_epoch={}\n".format(epoch + 1))
            f.write("train_loss={}\n".format(trainLoss))
            f.write("best_score_epoch={}\n".format(epoch + 1))
            f.write("best_score={}\n".format(self.bestScore))

        log.grey("Model weights saved at epoch {} with insample score={:.4f}, validation score={:.4f}, bestScore={:.4f}".format(epoch + 1, insampleScore, valScore, self.bestScore))
