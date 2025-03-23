import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from scipy.spatial import distance_matrix
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


root = "/z5s/cv/scan/ame"
dataDir = "/z5s/cps/LasV_fly/lasV_fly/Test"

# cfgname1 = "nyf.dss.cuda@3.batch@16.lr@40.loss@infoNCE" # 78.85
# cfgname2 = "nyf.dssentropy.cuda@1.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@2.ne" # 92.8
cfgname1 = "lvf.dss.cuda@1.batch@16.lr@40.loss@infoNCE" # 81.9 - 90.48
cfgname2 = "lvf.dssaccl.cuda@2.batch@16.lr@40.loss@infoNCEWeightedDirect.gridn@7.k@16.ne2" # 84.87 - 92.41

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
