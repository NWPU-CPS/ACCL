import os
import pathlib
from PIL import Image
import numpy as np
from misc.log import log
import torch
from factory.transformfactory import JupiterImageTransformFactory


class EvalDataSet(object):
    def __init__(self, qtensor, mtensor, rtensor, qpos, mpos, rpos, qFiles, gtFiles, rFiles, gtDistances):
        self.qtensor = qtensor
        self.mtensor = mtensor
        self.rtensor = rtensor
        self.qpos = qpos
        self.mpos = mpos
        self.rpos = rpos
        self.qFiles = qFiles
        self.gtFiles = gtFiles
        self.rFiles = rFiles
        self.gtDistances = gtDistances


class JupiterRawDataset():
    @classmethod
    def getRawDataLoader(cls, dcfg, entry, adapter):
        return JupiterRawDataset(
            name=entry.name,
            imageSize=dcfg.imageSize,
            queryDir=entry.queryDir,
            referenceDir=entry.referenceDir,
            matchFile=entry.matchFile,
            queryPosFile=entry.queryPosFile,
            referencePosFile=entry.referencePosFile,
            transformSet=dcfg.transformSet,
            adapter=adapter,
        )

    def __init__(self, name, imageSize, queryDir, referenceDir, matchFile, queryPosFile, referencePosFile, transformSet, adapter):
        self.name = name
        self.queryDir = queryDir
        self.referenceDir = referenceDir
        self.matchFile = matchFile
        self.queryPosFile = queryPosFile
        self.referencePosFile = referencePosFile
        self.transform = JupiterImageTransformFactory.getEvalTransformSet(transformSet, imageSize)
        self.adapter = adapter

        # load all images and filter unused
        queryImages = self.walkDir(self.queryDir)
        referenceImages = self.walkDir(self.referenceDir)
        self.qrmapDf = self.adapter.readMatchFile(self.matchFile)
        self.qrmapDf = self.qrmapDf.loc[self.qrmapDf["qname"].isin(queryImages)]

        self.queryImages = list(self.qrmapDf.loc[self.qrmapDf["qname"].isin(queryImages), "qname"].drop_duplicates())
        if len(self.queryImages) == 0:
            raise ValueError("[{}] No query images loaded from {}".format(self.name, queryDir))

        self.referenceImages = list(self.qrmapDf.loc[self.qrmapDf["rname"].isin(referenceImages), "rname"].drop_duplicates())
        if len(self.referenceImages) == 0:
            raise ValueError("[{}] No reference images loaded from {}".format(self.name, referenceDir))

        log.inf("[{}] raw-dataset initialized with <Q-{}, R-{}>".format(self.name, len(self.queryImages), len(self.referenceImages)))

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

    def collate(self, imageList, positionList):
        imageTensor = torch.stack(imageList)
        posTensor = torch.stack([torch.tensor(x) for x in positionList])
        return imageTensor, posTensor

    def load(self):
        log.inf("[{}] raw dataset loading all in once...".format(self.name))

        qposDf = self.adapter.readPositionFile(self.queryPosFile)
        rposDf = self.adapter.readPositionFile(self.referencePosFile)
        qrmap = {q : (r, d) for q, r, d in zip(self.qrmapDf["qname"], self.qrmapDf["rname"], self.qrmapDf["distance"])}

        qImages = []; qFiles = []; qPositions = []; gtDistances = []
        mImages = []; gtFiles = []; mPositions = []
        rImages = []; rFiles = []; rPositions = []

        for queryImageFile in self.queryImages:
            if queryImageFile not in qrmap:
                log.yellow("[{}] Failed to find query image in qrmap: {}, image skipped".format(self.name, queryImageFile))
                continue
            mappedRefImageFile, gtDistance = qrmap[queryImageFile]

            qposSub = qposDf.loc[qposDf["name"] == queryImageFile]
            if qposSub.shape[0] == 0:
                log.yellow("[{}] Failed to find query position info: {} in {}".format(self.name, queryImageFile, self.queryPosFile))
                continue

            mposSub = rposDf.loc[rposDf["name"] == mappedRefImageFile]
            if mposSub.shape[0] == 0:
                log.yellow("[{}] Failed to find reference position info: {} in {}".format(self.name, mappedRefImageFile, self.referencePosFile))
                import pdb; pdb.set_trace()
                continue

            qpos = (qposSub.iloc[0]["x"], qposSub.iloc[0]["y"])
            mpos = (mposSub.iloc[0]["x"], mposSub.iloc[0]["y"])
            qImage = self.loadRGBImage(os.path.join(self.queryDir, queryImageFile))
            mImage = self.loadRGBImage(os.path.join(self.referenceDir, mappedRefImageFile))

            qImages.append(qImage)
            qFiles.append(queryImageFile)
            qPositions.append(qpos)
            gtDistances.append(int(np.ceil(gtDistance)))

            mImages.append(mImage)
            gtFiles.append(mappedRefImageFile)
            mPositions.append(mpos)

        for refImageFile in self.referenceImages:
            rposSub = rposDf.loc[rposDf["name"] == refImageFile]
            if rposSub.shape[0] == 0:
                log.yellow("[{}] Failed to find reference position info: {} in {}".format(self.name, refImageFile, self.referencePosFile))
                continue
            rImage = self.loadRGBImage(os.path.join(self.referenceDir, refImageFile))
            rpos = (rposSub.iloc[0]["x"], rposSub.iloc[0]["y"])
            rImages.append(rImage)
            rFiles.append(refImageFile)
            rPositions.append(rpos)

        qtensor, qposTensor = self.collate(qImages, qPositions)
        mtensor, mposTensor = self.collate(mImages, mPositions)
        rtensor, rposTensor = self.collate(rImages, rPositions)

        return EvalDataSet(
            qtensor=qtensor,
            mtensor=mtensor,
            rtensor=rtensor,
            qpos=qposTensor,
            mpos=mposTensor,
            rpos=rposTensor,
            qFiles=qFiles,
            gtFiles=gtFiles,
            rFiles=rFiles,
            gtDistances=gtDistances,
        )

    def loadRGBImage(self, filename):
        image = Image.open(filename)
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return self.transform(image)
        else:
            image = image.convert("RGB")
            return self.transform(image)
