class JupyterDataLoaderFactory(object):
    @classmethod
    def getTrainDataLoader(cls, dcfg, entry):
        if entry.name == "nyf.triplet":
            from dl.nyf import JupiterNyfTripletDataset
            return JupiterNyfTripletDataset.getTrainDataLoader(dcfg, entry)
        if entry.name == "nyf.dss":
            from dl.nyf import JupiterNyfDssDataset
            return JupiterNyfDssDataset.getTrainDataLoader(dcfg, entry)
        if entry.name == "nyf.dssaccl":
            from dl.nyf import JupiterNyfDssAcclDataset
            return JupiterNyfDssAcclDataset.getTrainDataLoader(dcfg, entry)
        if entry.name == "nyf.dssentropy":
            from dl.nyf import JupiterNyfDssEntropyDataset
            return JupiterNyfDssEntropyDataset.getTrainDataLoader(dcfg, entry)

        if entry.name == "hwf.triplet":
            from dl.hwf import JupiterHwfTripletDataset
            return JupiterHwfTripletDataset.getTrainDataLoader(dcfg, entry)
        if entry.name == "hwf.dss":
            from dl.hwf import JupiterHwfDssDataset
            return JupiterHwfDssDataset.getTrainDataLoader(dcfg, entry)
        if entry.name == "hwf.dssaccl":
            from dl.hwf import JupiterHwfDssAcclDataset
            return JupiterHwfDssAcclDataset.getTrainDataLoader(dcfg, entry)
        if entry.name == "hwf.dssentropy":
            from dl.hwf import JupiterHwfDssEntropyDataset
            return JupiterHwfDssEntropyDataset.getTrainDataLoader(dcfg, entry)

        if entry.name == "lvf.triplet":
            from dl.lvf import JupiterLvfTripletDataset
            return JupiterLvfTripletDataset.getTrainDataLoader(dcfg, entry)
        if entry.name == "lvf.dss":
            from dl.lvf import JupiterLvfDssDataset
            return JupiterLvfDssDataset.getTrainDataLoader(dcfg, entry)
        if entry.name == "lvf.dssaccl":
            from dl.lvf import JupiterLvfDssAcclDataset
            return JupiterLvfDssAcclDataset.getTrainDataLoader(dcfg, entry)
        if entry.name == "lvf.dssentropy":
            from dl.lvf import JupiterLvfDssEntropyDataset
            return JupiterLvfDssEntropyDataset.getTrainDataLoader(dcfg, entry)

        raise ValueError("DataLoader not registered: {}".format(entry))

    @classmethod
    def getEvalDataLoader(cls, dcfg, entry):
        if entry.name == "nyf.raw":
            from dl.nyf import JupiterNyfRawDataset
            return JupiterNyfRawDataset.getRawDataLoader(dcfg, entry)
        if entry.name == "hwf.raw":
            from dl.hwf import JupiterHwfRawDataset
            return JupiterHwfRawDataset.getRawDataLoader(dcfg, entry)
        if entry.name == "lvf.raw":
            from dl.lvf import JupiterLvfRawDataset
            return JupiterLvfRawDataset.getRawDataLoader(dcfg, entry)
        raise ValueError("DataLoader not registered: {}".format(entry))
