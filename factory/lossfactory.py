from misc.log import log


class JupyterLossFactory(object):
    @classmethod
    def initLoss(cls, mcfg):
        match mcfg.lossName:
            case "triplet":
                from train.tripletloss import SyntheticLoss
                return SyntheticLoss(mcfg)
            case "infoNCE":
                from train.infonceloss import InfoNCE
                return InfoNCE(mcfg)
            case "infoNCEWeighted":
                from train.infonceloss import InfoNCEWeighted
                return InfoNCEWeighted(mcfg)
            case "infoNCEWeightedDirect":
                from train.infonceloss import InfoNCEWeightedDirect
                return InfoNCEWeightedDirect(mcfg)
            case "infoNCEWeightedPlus":
                from train.infonceloss import InfoNCEWeightedPlus
                return InfoNCEWeightedPlus(mcfg)
            case other:
                raise ValueError("Loss not reigstered: {}".format(mcfg.lossName))
