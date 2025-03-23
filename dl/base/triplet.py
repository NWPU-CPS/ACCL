import os
import math
import pathlib
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from misc.log import log
from factory.transformfactory import JupiterImageTransformFactory


class TripletItem(object):
    def __init__(self, queryImageFile, refImageFile, negImageFile, queryImage, refImage, negImage):
        self.queryImageFile = queryImageFile
        self.refImageFile = refImageFile
        self.negImageFile = negImageFile
        self.queryImage = queryImage
        self.refImage = refImage
        self.negImage = negImage


class JupiterTripletDataset(Dataset):
    @staticmethod
    def trainCollate(batch):
        queryImages = []
        refImages = []
        negImages = []

        for item in batch:
            queryImages.append(item.queryImage)
            refImages.append(item.refImage)
            negImages.append(item.negImage)

        qtensor = torch.stack(queryImages)
        rtensor = torch.stack(refImages)
        ntensor = torch.stack(negImages)

        return qtensor, rtensor, ntensor

    @classmethod
    def getTrainDataLoader(cls, dcfg, entry, adapter):
        dataset = cls(
            name=entry.name,
            imageSize=dcfg.imageSize,
            queryDir=entry.queryDir,
            referenceDir=entry.referenceDir,
            matchFile=entry.matchFile,
            queryPosFile=entry.queryPosFile,
            referencePosFile=entry.referencePosFile,
            ntriplets=entry.ntriplets,
            dcore=min(100, dcfg.dcore),
            transformSet=dcfg.transformSet,
            resampleMode=dcfg.resampleMode,
            adapter=adapter,
        )
        loader = DataLoader(
            dataset,
            batch_size=dcfg.batchSize,
            num_workers=min(10, dcfg.dcore),
            pin_memory=True,
            drop_last=True,
            collate_fn=cls.trainCollate,
        )
        return loader

    def __init__(self, name, imageSize, queryDir, referenceDir, matchFile, queryPosFile, referencePosFile, ntriplets, dcore, transformSet, resampleMode, adapter):
        self.name = name
        self.imageSize = imageSize
        self.queryDir = queryDir
        self.referenceDir = referenceDir
        self.matchFile = matchFile
        self.queryPosFile = queryPosFile
        self.referencePosFile = referencePosFile
        self.ntriplets = ntriplets
        self.dcore = dcore
        self.resampleMode = resampleMode
        self.adapter = adapter
        self.trainTransform = JupiterImageTransformFactory.getTrainTransformSet(transformSet, imageSize)
        self.evalTransform = JupiterImageTransformFactory.getEvalTransformSet(transformSet, imageSize)

        # load all images and filter unused
        queryImages = self.walkDir(self.queryDir)
        referenceImages = self.walkDir(self.referenceDir)
        self.qrmapDf = self.adapter.readMatchFile(self.matchFile)
        self.qrmapDf = self.qrmapDf.loc[self.qrmapDf["qname"].isin(queryImages)]
        self.qrmap = {q : r for q, r in zip(self.qrmapDf["qname"], self.qrmapDf["rname"])}

        self.queryImages = list(self.qrmapDf.loc[self.qrmapDf["qname"].isin(queryImages), "qname"].drop_duplicates())
        if len(self.queryImages) == 0:
            raise ValueError("[{}] No query images loaded from {}".format(self.name, queryDir))

        self.referenceImages = list(self.qrmapDf.loc[self.qrmapDf["rname"].isin(referenceImages), "rname"].drop_duplicates())
        if len(self.referenceImages) == 0:
            raise ValueError("[{}] No reference images loaded from {}".format(self.name, referenceDir))

        # build distance - query map
        self.qrmapDf.loc[:, "roundDist"] = self.qrmapDf["distance"].apply(np.ceil).astype(int)
        self.distanceQueryMap = {}
        for dist in self.qrmapDf["roundDist"].unique():
            sub = self.qrmapDf.loc[self.qrmapDf["roundDist"] == dist]
            self.distanceQueryMap[dist] = list(sub["qname"])
        self.maxDistance = self.qrmapDf["roundDist"].max()

        # load position info and calc dist mat
        self.qposDf = self.adapter.readPositionFile(self.queryPosFile)
        self.rposDf = self.adapter.readPositionFile(self.referencePosFile)
        self.qposDf = self.qposDf.loc[self.qposDf["name"].isin(queryImages) & self.qposDf["name"].isin(self.qrmapDf["qname"])]
        self.rposDf = self.rposDf.loc[self.rposDf["name"].isin(referenceImages) & self.rposDf["name"].isin(self.qrmapDf["rname"])]
        qposTensor = torch.tensor(self.qposDf[["x", "y"]].to_numpy(), dtype=torch.float32)
        rposTensor = torch.tensor(self.rposDf[["x", "y"]].to_numpy(), dtype=torch.float32)
        self.qrDistMat = torch.cdist(qposTensor, rposTensor).numpy()
        self.qrDistSortedIndexMat = np.argsort(self.qrDistMat, axis=1)
        self.qindexVec = list(self.qrmapDf["qindex"])
        self.rindexVec = list(self.qrmapDf["rindex"].unique())
        self.qimap = {q : qi for q, qi in zip(self.queryImages, self.qindexVec)}
        self.rimap = {r : ri for r, ri in zip(self.referenceImages, self.rindexVec)}

        self.triplets = self.randomResample()
        if len(self.triplets) == 0:
            raise ValueError("[{}] No triplets loaded".format(self.name))

        log.inf("[{}] triplet-dataset initialized with <Q-{}, R-{}, T-{}>".format(self.name, len(self.queryImages), len(self.referenceImages), len(self.triplets)))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]
        item.queryImage = self.loadImage(os.path.join(self.queryDir, item.queryImageFile), eval=False)
        item.refImage = self.loadImage(os.path.join(self.referenceDir, item.refImageFile), eval=True)
        item.negImage = self.loadImage(os.path.join(self.referenceDir, item.negImageFile), eval=True)
        return item

    def walkDir(self, dirpath):
        images = []
        for root, _, files in os.walk(dirpath):
            for file in files:
                if pathlib.Path(file).suffix not in self.adapter.suffix:
                    continue
                abspath = os.path.join(root, file)
                relpath = os.path.relpath(abspath, dirpath)
                images.append(relpath)
        return sorted(images)

    def generateTriplet(self, queryImageFile, refImageFile, negImageFile):
        return TripletItem(
            queryImageFile=queryImageFile,
            refImageFile=refImageFile,
            negImageFile=negImageFile,
            queryImage=None,
            refImage=None,
            negImage=None,
        )

    def sampleQueryImages(self, queryImages, nsamples):
        sampledQueryImages = []
        if nsamples <= len(queryImages):
            sampledQueryImages = random.sample(queryImages, nsamples)
        else:
            rounds = int(nsamples / len(queryImages))
            sampledQueryImages = queryImages * rounds + random.sample(queryImages, nsamples % len(queryImages))
        return sampledQueryImages

    def sampleNegativeRandom(self, queryImageFile, refImageFile):
        negImageFile = random.sample(self.referenceImages, 1)[0]
        while negImageFile == refImageFile:
            negImageFile = random.sample(self.referenceImages, 1)[0]
        return negImageFile

    def precision2distance(self, precision, minDistance, maxDistance):
        precision = 1 - max(0, min(1, precision))
        val = 0.5 * (1 + math.cos(math.pi * precision))
        dist = int(minDistance + (maxDistance - minDistance) * val)
        return max(minDistance, min(dist, maxDistance))

    def sampleNegativeStrat(self, queryImageFile, refImageFile, distance, precision):
        negDistance = self.precision2distance(precision, distance, self.maxDistance * 10)
        qindex = self.qimap[queryImageFile]
        rdistRow = self.qrDistMat[qindex]
        rdistSortedIndexRow = self.qrDistSortedIndexMat[qindex]
        for ri in rdistSortedIndexRow:
            dist = rdistRow[ri]
            selectedRefImage = self.referenceImages[ri]
            if selectedRefImage == refImageFile: # skip the ground truth
                continue
            if dist >= negDistance:
                return selectedRefImage
        # should not reach this line
        import pdb; pdb.set_trace()

    def resample(self, epoch, insampleEvalDf):
        log.inf("[{}] Resampling with method {}...".format(self.name, self.resampleMode))
        if self.resampleMode == "switch":
            if epoch % 2 == 0:
                self.triplets = self.randomResample()
            else:
                self.triplets = self.precisionResample(insampleEvalDf)
        elif self.resampleMode == "precision":
            self.triplets = self.precisionResample(insampleEvalDf)
        else: # random
            self.triplets = self.randomResample()

    def randomResample(self):
        log.inf("[{}] Generating {} triplets...".format(self.name, self.ntriplets))
        sampledQueryImages = self.sampleQueryImages(self.queryImages, self.ntriplets)
        triplets = []
        for queryImageFile in sampledQueryImages:
            if queryImageFile not in self.qrmap:
                log.yellow("[{}] Failed to find query image in qrmap: {}, image skipped".format(self.name, queryImageFile))
                continue
            refImageFile = self.qrmap[queryImageFile]
            negImageFile = self.sampleNegativeRandom(queryImageFile, refImageFile)
            triplet = self.generateTriplet(queryImageFile, refImageFile, negImageFile)
            triplets.append(triplet)
        return triplets

    def precisionResample(self, insampleEvalDf):
        ozdf = insampleEvalDf.loc[insampleEvalDf["order"] == 0].copy()
        ozdf.loc[:, "precision"] = ozdf["T"] / ozdf["N"]
        ozdf.loc[ozdf["N"] < 10, "precision"] = ozdf["N"] / ozdf["N"].sum()
        ozdf.loc[:, "normp"] = ozdf["precision"] / ozdf["precision"].sum()
        log.inf("[{}] Resampling {} triplets with distribution...\n{}".format(self.name, self.ntriplets, ozdf))
        try:
            ozdf.loc[:, "levelN"] = (ozdf["normp"] * self.ntriplets).apply(np.ceil).astype(int)
        except:
            log.red("ERROR")
            import pdb; pdb.set_trace()
        triplets = []
        for distance, precision, levelN in zip(ozdf["distance"], ozdf["precision"], ozdf["levelN"]):
            subN = min(levelN, self.ntriplets - len(triplets))
            if subN == 0:
                break
            subQueryImages = self.distanceQueryMap[distance]
            subSampledQueryImages = self.sampleQueryImages(subQueryImages, subN)
            for queryImageFile in subSampledQueryImages:
                if queryImageFile not in self.qrmap:
                    log.yellow("[{}] Failed to find query image in qrmap: {}, image skipped".format(self.name, queryImageFile))
                    continue
                refImageFile = self.qrmap[queryImageFile]
                negImageFile = self.sampleNegativeStrat(queryImageFile, refImageFile, distance, precision)
                triplet = self.generateTriplet(queryImageFile, refImageFile, negImageFile)
                triplets.append(triplet)
        return triplets

    def loadImage(self, filename, eval=False):
        transformSet = self.evalTransform if eval else self.trainTransform
        image = Image.open(filename)
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return transformSet(image)
        else:
            image = image.convert("RGB")
            return transformSet(image)
