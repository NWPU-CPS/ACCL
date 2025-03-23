import pandas as pd


class LvfAdapter(object):
    def __init__(self):
        self.suffix = ".jpg"
        self.geoMode = "cartesian"
        self.rDelta = 15
        # self.eps = 0
        self.eps = 5

    def readMatchFile(self, matchFile):
        df = pd.read_csv(matchFile).rename(columns={
            "query_ind": "qindex",
            "query_name": "qname",
            "ref_ind": "rindex",
            "ref_name": "rname",
        })
        return df

    def readPositionFile(self, posFile):
        df = pd.read_csv(posFile)
        df = df.rename(columns={"easting": "x", "northing": "y"})
        return df[["name", "x", "y"]].copy()


class JupiterLvfDssDataset(object):
    @staticmethod
    def getTrainDataLoader(dcfg, entry):
        from dl.base.dss import JupiterDssDataset
        return JupiterDssDataset.getTrainDataLoader(dcfg=dcfg, entry=entry, adapter=LvfAdapter())


class JupiterLvfDssAcclDataset(object):
    @staticmethod
    def getTrainDataLoader(dcfg, entry):
        from dl.base.dssaccl import JupiterDssAcclDataset
        return JupiterDssAcclDataset.getTrainDataLoader(dcfg=dcfg, entry=entry, adapter=LvfAdapter())


class JupiterLvfDssEntropyDataset(object):
    @staticmethod
    def getTrainDataLoader(dcfg, entry):
        from dl.base.dssentropy import JupiterDssEntropyDataset
        return JupiterDssEntropyDataset.getTrainDataLoader(dcfg=dcfg, entry=entry, adapter=LvfAdapter())


class JupiterLvfRawDataset(object):
    @staticmethod
    def getRawDataLoader(dcfg, entry):
        from dl.base.raw import JupiterRawDataset
        return JupiterRawDataset.getRawDataLoader(dcfg=dcfg, entry=entry, adapter=LvfAdapter())


class JupiterLvfTripletDataset(object):
    @staticmethod
    def getTrainDataLoader(dcfg, entry):
        from dl.base.triplet import JupiterTripletDataset
        return JupiterTripletDataset.getTrainDataLoader(dcfg=dcfg, entry=entry, adapter=LvfAdapter())
