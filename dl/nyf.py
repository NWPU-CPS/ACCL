import pandas as pd


class HyfAdapter(object):
    def __init__(self):
        self.suffix = ".jpg"
        self.geoMode = "haversine"
        self.rDelta = 4.78 # distance between connected reference images in meters
        self.eps = 5
        # self.eps = 0.5

    def readMatchFile(self, matchFile):
        df = pd.read_csv(matchFile, names=["qindex", "qname", "rindex", "rname", "distance"])
        return df

    def readPositionFile(self, posFile):
        df = pd.read_csv(posFile)
        df = df.rename(columns={"lat": "x", "long": "y"})
        return df[["name", "x", "y"]].copy()


class JupiterNyfDssDataset(object):
    @staticmethod
    def getTrainDataLoader(dcfg, entry):
        from dl.base.dss import JupiterDssDataset
        return JupiterDssDataset.getTrainDataLoader(dcfg=dcfg, entry=entry, adapter=HyfAdapter())


class JupiterNyfDssAcclDataset(object):
    @staticmethod
    def getTrainDataLoader(dcfg, entry):
        from dl.base.dssaccl import JupiterDssAcclDataset
        return JupiterDssAcclDataset.getTrainDataLoader(dcfg=dcfg, entry=entry, adapter=HyfAdapter())


class JupiterNyfDssEntropyDataset(object):
    @staticmethod
    def getTrainDataLoader(dcfg, entry):
        from dl.base.dssentropy import JupiterDssEntropyDataset
        return JupiterDssEntropyDataset.getTrainDataLoader(dcfg=dcfg, entry=entry, adapter=HyfAdapter())


class JupiterNyfRawDataset(object):
    @staticmethod
    def getRawDataLoader(dcfg, entry):
        from dl.base.raw import JupiterRawDataset
        return JupiterRawDataset.getRawDataLoader(dcfg=dcfg, entry=entry, adapter=HyfAdapter())


class JupiterNyfTripletDataset(object):
    @staticmethod
    def getTrainDataLoader(dcfg, entry):
        from dl.base.triplet import JupiterTripletDataset
        return JupiterTripletDataset.getTrainDataLoader(dcfg=dcfg, entry=entry, adapter=HyfAdapter())
