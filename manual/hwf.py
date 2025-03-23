import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from scipy.spatial import distance_matrix
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


root = "/z5s/cv/scan/ame"
dataDir = "/z5s/cps/Hollywood_Fly/hollywood_fly/Test"
rootDir = "/z5s/cps/Hollywood_Fly/hollywood_fly"

cfgname1 = "hwf.dss.cuda@1.batch@16.lr@40.loss@infoNCE"
cfgname2 = "hwf.dssaccl.cuda@0.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@2.ne"

df1 = pd.read_csv(os.path.join(root, cfgname1, "eval/detail.csv"))
df2 = pd.read_csv(os.path.join(root, cfgname2, "eval/detail.csv"))

qposDf = pd.read_csv(os.path.join(dataDir, "query.csv"))
rposDf = pd.read_csv(os.path.join(dataDir, "reference.csv"))

df1 = df1.merge(qposDf[["name", "easting", "northing"]].rename(columns={"name": "qFiles", "easting": "qx", "northing": "qy"}), on="qFiles", how="left")
df1 = df1.merge(rposDf[["name", "easting", "northing"]].rename(columns={"name": "mFiles", "easting": "mx", "northing": "my"}), on="mFiles", how="left")

df2 = df2.merge(qposDf[["name", "easting", "northing"]].rename(columns={"name": "qFiles", "easting": "qx", "northing": "qy"}), on="qFiles", how="left")
df2 = df2.merge(rposDf[["name", "easting", "northing"]].rename(columns={"name": "mFiles", "easting": "mx", "northing": "my"}), on="mFiles", how="left")

def calcDist(parray1, parray2):
    geoMat = distance_matrix(parray1, parray2)
    # geoMat = haversine_distances(parray1, parray2) * 6387209.7
    return np.diagonal(geoMat)

df1.loc[:, "d"] = calcDist(df1[["qx", "qy"]], df1[["mx", "my"]])
df2.loc[:, "d"] = calcDist(df2[["qx", "qy"]], df2[["mx", "my"]])


import pdb; pdb.set_trace()
