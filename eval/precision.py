import torch
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree


class JupiterPrecisionEvaluator(object):
    def __init__(self, mcfg):
        self.mcfg = mcfg

    def increaseMapCount(self, distanceMap, distance):
        if distance in distanceMap:
            distanceMap[distance] += 1
        else:
            distanceMap[distance] = 1

    def modelInference(self, model, tensor):
        outList = []
        for i in range(0, tensor.shape[0], self.mcfg.batchSize):
            subout = model(tensor[i : min(tensor.shape[0], i + self.mcfg.batchSize)].to(self.mcfg.device))
            outList.append(subout.cpu())
        return torch.cat(outList, dim=0)

    def eval(self, model, evalData):
        qtensor = evalData.qtensor
        rtensor = evalData.rtensor

        qout = self.modelInference(model, qtensor)
        rout = self.modelInference(model, rtensor)

        distanceMapN = {}
        distanceMapTList = []
        for i in range(self.mcfg.kNeighbors):
            distanceMapTList.append({})

        refTree = KDTree(rout.cpu())
        _, distanceMat = refTree.query(qout.cpu(), k=self.mcfg.kNeighbors)
        qFiles = []
        mFiles = []
        gtFiles = []
        aps = []

        for i in range(qout.shape[0]):
            matchedRefIndexVec = distanceMat[i]
            gtDistance = evalData.gtDistances[i]
            self.increaseMapCount(distanceMapN, gtDistance)
            qFiles.append(evalData.qFiles[i])
            mFiles.append(evalData.rFiles[matchedRefIndexVec[0]])
            gtFiles.append(evalData.gtFiles[i])

            matched = False
            for orderIdx, matchedRefIndex in enumerate(matchedRefIndexVec):
                if evalData.gtFiles[i] == evalData.rFiles[matchedRefIndex]:
                    for j in range(orderIdx, self.mcfg.kNeighbors):
                        self.increaseMapCount(distanceMapTList[j], gtDistance)
                    aps.append(1 / (orderIdx + 1))
                    matched = True
                    break

            if not matched:
                aps.append(np.nan)

        uniqueDistanceVec = sorted(list(set(evalData.gtDistances)))
        samplesCountVec = []
        for distance in uniqueDistanceVec:
            if distance in distanceMapN:
                samplesCountVec.append(distanceMapN[distance])
            else:
                samplesCountVec.append(0)

        ndf = pd.DataFrame({
            "distance": uniqueDistanceVec,
            "N": samplesCountVec,
        })
        tdfList = []
        for i in range(self.mcfg.kNeighbors):
            trueCountVec = []
            for distance in uniqueDistanceVec:
                distanceMapT = distanceMapTList[i]
                if distance in distanceMapT:
                    trueCountVec.append(distanceMapT[distance])
                else:
                    trueCountVec.append(0)
            tdf = pd.DataFrame({
                "distance": uniqueDistanceVec,
                "order": i,
                "T": trueCountVec,
            })
            tdfList.append(tdf)

        tdf = pd.concat(tdfList)
        evalDf = tdf.merge(ndf, on="distance", how="left")

        smrDf = evalDf.groupby("order")[["T", "N"]].sum().reset_index()
        smrDf.loc[:, "precision"] = smrDf["T"] / smrDf["N"]

        detailedDf = pd.DataFrame({
            "qFiles": qFiles,
            "mFiles": mFiles,
            "gtFiles": gtFiles,
            "aps": aps,
        })

        return evalDf, smrDf, detailedDf
