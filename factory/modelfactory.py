from misc.log import log


class JupyterModelFactory(object):
    @classmethod
    def loadModelClass(cls, mcfg):
        if mcfg.modelName == "vit":
            from model.vitmodel import VitModel
            return VitModel
        if mcfg.modelName == "timm":
            from model.timm import TimmModel
            return TimmModel
        if mcfg.modelName == "swin":
            from model.ext.swin import SwinTransformer
            return SwinTransformer
        if mcfg.modelName == "vit2":
            from model.ext.vit2 import ViT
            return ViT
        raise ValueError("Model not registered: {}".format(mcfg))

    @classmethod
    def createTrainModel(cls, mcfg):
        return cls.loadModelClass(mcfg).createTrainModel(mcfg)

    @classmethod
    def createEvalModel(cls, mcfg, modelFile):
        return cls.loadModelClass(mcfg).createEvalModel(mcfg, modelFile)
