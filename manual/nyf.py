import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


root = "/z5s/cv/scan/ame.back2"
dataDir = "/z5s/cv/cvdata/NewYorkFly/Test"

cfgname1 = "nyf.dss.cuda@3.batch@16.lr@40.loss@infoNCE" # 78.85
cfgname2 = "nyf.dssentropy.cuda@1.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@2.ne" # 92.8

df1 = pd.read_csv(os.path.join(root, cfgname1, "eval/detail.csv"))
df2 = pd.read_csv(os.path.join(root, cfgname2, "eval/detail.csv"))

qposDf = pd.read_csv(os.path.join(dataDir, "query.csv"))
rposDf = pd.read_csv(os.path.join(dataDir, "reference.csv"))

df1 = df1.merge(qposDf[["name", "lat", "long"]].rename(columns={"name": "qFiles", "lat": "qx", "long": "qy"}), on="qFiles", how="left")
df1 = df1.merge(rposDf[["name", "lat", "long"]].rename(columns={"name": "mFiles", "lat": "mx", "long": "my"}), on="mFiles", how="left")

df2 = df2.merge(qposDf[["name", "lat", "long"]].rename(columns={"name": "qFiles", "lat": "qx", "long": "qy"}), on="qFiles", how="left")
df2 = df2.merge(rposDf[["name", "lat", "long"]].rename(columns={"name": "mFiles", "lat": "mx", "long": "my"}), on="mFiles", how="left")

def calcDist(parray1, parray2):
    parray1 = np.radians(parray1)
    parray2 = np.radians(parray2)
    geoMat = haversine_distances(parray1, parray2) * 6387209.7
    return np.diagonal(geoMat)

df1.loc[:, "d"] = calcDist(df1[["qx", "qy"]], df1[["mx", "my"]])
df2.loc[:, "d"] = calcDist(df2[["qx", "qy"]], df2[["mx", "my"]])


import pdb; pdb.set_trace()
