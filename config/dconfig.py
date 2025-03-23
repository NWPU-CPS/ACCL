class DataTripletConfigEntry(object):
    def __init__(self):
        self.name = None
        self.queryDir = None
        self.referenceDir = None
        self.matchFile = None
        self.queryPosFile = None
        self.referencePosFile = None

        self.ntriplets = None

    def finalize(self):
        if self.queryDir is None:
            raise ValueError("Query directory not set")
        if self.referenceDir is None:
            raise ValueError("Reference directory not set")
        if self.matchFile is None:
            raise ValueError("MatchFile not set")
        if self.queryPosFile is None:
            raise ValueError("QueryPosFile not set")
        if self.referencePosFile is None:
            raise ValueError("ReferencePosFile not set")
        return self


class DataDssConfigEntry(object):
    def __init__(self):
        self.name = None
        self.queryDir = None
        self.referenceDir = None
        self.matchFile = None
        self.queryPosFile = None
        self.referencePosFile = None

        self.gridN = 4
        self.kNeighbors = 64
        self.neighborsRange = 128

    def finalize(self):
        if self.queryDir is None:
            raise ValueError("Query directory not set")
        if self.referenceDir is None:
            raise ValueError("Reference directory not set")
        if self.matchFile is None:
            raise ValueError("MatchFile not set")
        if self.queryPosFile is None:
            raise ValueError("QueryPosFile not set")
        if self.referencePosFile is None:
            raise ValueError("ReferencePosFile not set")
        return self


class DataConfig(object):
    def __init__(self):
        self.imageDir = None
        self.dcore = 20
        self.transformSet = "simple"
        self.resampleMode = "switch"

        self.trainEntry = None
        self.validateEntry = None
        self.insampleEvalEntry = None
        self.evalEntry = None

        # enriched
        self.imageSize = None
        self.batchSize = None
        self.seed = None
        self.device = None
        self.maxEpoch = None

    def enrich(self, tags):
        for tag in tags:
            tokens = tag.split("@")
            match tokens[0]:
                case "trans":
                    self.transformSet = tokens[1]
                case "resamplemode":
                    self.resampleMode = tokens[1]
                case "dk":
                    self.trainEntry.kNeighbors = [64, 48, 32, 16, 8]
                    self.trainEntry.neighborsRange = [128, 128, 64, 64, 32]
                case "k":
                    self.trainEntry.kNeighbors = int(tokens[1])
                case "K":
                    self.trainEntry.neighborsRange = int(tokens[1])
                case "gridn":
                    self.trainEntry.gridN = int(tokens[1])
        return self

    def finalize(self):
        return self
