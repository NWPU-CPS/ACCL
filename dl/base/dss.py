import os
import pathlib
import math
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from misc.log import log
from sklearn.metrics.pairwise import haversine_distances
from factory.transformfactory import JupiterImageTransformFactory


class JupiterDssDataset(Dataset):
    @staticmethod
    def collate(batch):
        queryImages = []
        refImages = []
        distances = []
        for (queryImage, refImage, distance) in batch:
            queryImages.append(queryImage)
            refImages.append(refImage)
            distances.append(distance)
        qtensor = torch.stack(queryImages)
        rtensor = torch.stack(refImages)
        dtensor = torch.tensor(distances).type(torch.float32)
        return qtensor, rtensor, dtensor

    @classmethod
    def getTrainDataLoader(cls, dcfg, entry, adapter):
        dataset = cls(
            name=entry.name,
            queryDir=entry.queryDir,
            referenceDir=entry.referenceDir,
            matchFile=entry.matchFile,
            queryPosFile=entry.queryPosFile,
            referencePosFile=entry.referencePosFile,
            kNeighbors=entry.kNeighbors,
            neighborsRange=entry.neighborsRange,
            imageSize=dcfg.imageSize,
            transformSet=dcfg.transformSet,
            maxEpoch=dcfg.maxEpoch,
            adapter=adapter,
        )
        loader = DataLoader(
            dataset,
            batch_size=dcfg.batchSize,
            num_workers=dcfg.dcore,
            pin_memory=True,
            drop_last=True,
            collate_fn=cls.collate,
        )
        return loader

    def __init__(self, name, queryDir, referenceDir, matchFile, queryPosFile, referencePosFile, kNeighbors, neighborsRange, imageSize, transformSet, maxEpoch, adapter):
        self.name = name
        self.imageSize = imageSize
        self.queryDir = queryDir
        self.referenceDir = referenceDir
        self.matchFile = matchFile
        self.queryPosFile = queryPosFile
        self.referencePosFile = referencePosFile
        self.kNeighbors = kNeighbors
        self.neighborsRange = neighborsRange
        self.maxEpoch = maxEpoch
        self.adapter = adapter

        self.trainTransform = JupiterImageTransformFactory.getTrainTransformSet(transformSet, imageSize)
        self.evalTransform = JupiterImageTransformFactory.getEvalTransformSet(transformSet, imageSize)

        # load all images and filter unused
        queryImages = self.walkDir(self.queryDir)
        referenceImages = self.walkDir(self.referenceDir)
        self.qrmapDf = self.adapter.readMatchFile(self.matchFile)
        self.qrmapDf = self.qrmapDf.loc[self.qrmapDf["qname"].isin(queryImages)]
        self.qindexVec = list(self.qrmapDf["qindex"])
        self.rindexVec = list(self.qrmapDf["rindex"].unique())

        self.queryImages = list(self.qrmapDf["qname"].drop_duplicates())
        if len(self.queryImages) == 0:
            raise ValueError("[{}] No query images loaded from {}".format(self.name, queryDir))

        # load position info and calc dist mat
        self.qposDf = self.adapter.readPositionFile(self.queryPosFile)
        self.rposDf = self.adapter.readPositionFile(self.referencePosFile)
        self.qposDf = self.qposDf.loc[self.qposDf["name"].isin(queryImages) & self.qposDf["name"].isin(self.qrmapDf["qname"])]
        self.rposDf = self.rposDf.loc[self.rposDf["name"].isin(referenceImages) & self.rposDf["name"].isin(self.qrmapDf["rname"])]
        qposTensor = torch.tensor(self.qposDf[["x", "y"]].to_numpy(), dtype=torch.float64) # use float64 to avoid overflow problems
        rposTensor = torch.tensor(self.rposDf[["x", "y"]].to_numpy(), dtype=torch.float64)
        self.qrDistMat = self.geoMatrix(qposTensor, rposTensor).type(torch.float32)

        # multiple Qs could map to same R
        self.qrIndexMap = {qi : ri for qi, ri in zip(self.qrmapDf["qindex"], self.qrmapDf["rindex"])}
        self.rqIndexMap = {}
        for qindex, rindex in self.qrIndexMap.items():
            if rindex in self.rqIndexMap:
                self.rqIndexMap[rindex].append(qindex)
            else:
                self.rqIndexMap[rindex] = [qindex]

        self.queryImagesMap = {qi : qf for qi, qf in zip(self.qrmapDf["qindex"], self.qrmapDf["qname"])}
        self.queryDistanceMap = {qi : qf for qi, qf in zip(self.qrmapDf["qindex"], self.qrmapDf["distance"])}
        self.referenceImagesMap = {ri : rf for ri, rf in zip(self.qrmapDf["rindex"], self.qrmapDf["rname"])}

        self.shuffledQIndexes = self.shuffle(self.qrDistMat.max() - self.qrDistMat, self.getKNeighbors(0), self.getKRange(0))
        log.inf("[{}] dss-dataset initialized with <Q-{}, R-{}, k-{}, K-{}>".format(self.name, len(self.queryImagesMap), len(self.referenceImagesMap), self.kNeighbors, self.neighborsRange))

    def __len__(self):
        return len(self.shuffledQIndexes)

    def __getitem__(self, idx):
        qindex = self.shuffledQIndexes[idx]
        rindex = self.qrIndexMap[qindex]
        qFile = self.queryImagesMap[qindex]
        rFile = self.referenceImagesMap[rindex]
        distance = self.queryDistanceMap[qindex]
        qImage = self.loadImage(os.path.join(self.queryDir, qFile), eval=False)
        rImage = self.loadImage(os.path.join(self.referenceDir, rFile), eval=True)
        return qImage, rImage, distance

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

    def geoMatrix(self, parray1, parray2):
        match self.adapter.geoMode:
            case "cartesian":
                return torch.cdist(parray1, parray2)
            case "haversine":
                parray1 = np.radians(parray1)
                parray2 = np.radians(parray2)
                geoMat = haversine_distances(parray1, parray2) * 6387209.7
                return torch.tensor(geoMat)
            case other:
                raise ValueError("Invalid geoMode: {}".format(self.adapter.geoMode))

    def loadImage(self, filename, eval=False):
        transformSet = self.evalTransform if eval else self.trainTransform
        image = Image.open(filename)
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return transformSet(image)
        else:
            image = image.convert("RGB")
            return transformSet(image)

    def shuffle(self, similarityMat, kNeighbors, kRange):
        """
        similarityMat: Q x R matrix
        """
        queryIndexVec = np.arange(len(self.queryImages))
        np.random.shuffle(queryIndexVec)
        _, indexMat = torch.topk(similarityMat, kRange + 1, dim=1)
        indexMat = indexMat.numpy()
        shuffledIndexes = []

        qIndexSelected = set()
        for qi in queryIndexVec:
            qindex = self.qindexVec[qi]
            if qindex in qIndexSelected:
                continue
            qIndexSelected.add(qindex)
            shuffledIndexes.append(qindex)
            gtIndex = self.qrIndexMap[qindex]
            riRow = indexMat[qindex]

            neighbors = []
            for ri in riRow:
                rindex = self.rindexVec[ri]
                if rindex == gtIndex:
                    continue
                mappedQIndexes = [x for x in self.rqIndexMap[rindex] if x not in qIndexSelected]
                neighbors.extend(mappedQIndexes)
                if len(neighbors) >= kRange:
                    break

            halfK = int(kNeighbors / 2)
            nearestHalfK = neighbors[0 : halfK - 1]
            restNeighbors = neighbors[halfK - 1:]
            randomHalfK = list(np.random.choice(restNeighbors, min(len(restNeighbors), halfK), replace=False))
            qIndexSelected.update(nearestHalfK + randomHalfK)
            shuffledIndexes.extend(nearestHalfK + randomHalfK)

        return shuffledIndexes

    def getKNeighbors(self, epoch):
        if not isinstance(self.kNeighbors, list) and not isinstance(self.kNeighbors, tuple):
            return self.kNeighbors
        index = max(0, math.ceil(epoch / self.maxEpoch * len(self.kNeighbors)) - 1)
        return self.kNeighbors[index]

    def getKRange(self, epoch):
        if not isinstance(self.neighborsRange, list) and not isinstance(self.neighborsRange, tuple):
            return self.neighborsRange
        index = max(0, math.ceil(epoch / self.maxEpoch * len(self.neighborsRange)) - 1)
        return self.neighborsRange[index]

    def cosineSimilarityMatrix(self, qout, rout):
        similarityMat = qout @ rout.T
        return similarityMat

    def reshuffle(self, epoch, qout, rout):
        log.inf("[{}] Reshuffling...".format(self.name))
        similarityMat = self.cosineSimilarityMatrix(qout, rout)
        self.shuffledQIndexes = self.shuffle(similarityMat, self.getKNeighbors(epoch), self.getKRange(epoch))
