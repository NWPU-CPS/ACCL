import torch
import timm
import numpy as np
import torch.nn as nn
from misc.log import log


class TimmModel(nn.Module):
    @classmethod
    def createTrainModel(cls, mcfg):
        model = cls(model_name=mcfg.phase, img_size=mcfg.inputShape)
        model.setInferenceMode(False)
        model.train()
        return model

    @classmethod
    def createEvalModel(cls, mcfg, modelFile):
        model = cls(model_name=mcfg.phase, img_size=mcfg.inputShape)
        model.load(modelFile)
        model.setInferenceMode(True)
        model.eval()
        return model

    def __init__(self, model_name, img_size, checkpoint=None):
        super(TimmModel, self).__init__()
        self.img_size = img_size
        pretrained = checkpoint is None

        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size, checkpoint_path=checkpoint)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, checkpoint_path=checkpoint)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.inferenceMode = False
        self.backboneFreezed = False

    def setFreezeBackbone(self, flag):
        self.backboneFreezed = flag

    def setInferenceMode(self, flag):
        self.inferenceMode = flag

    def forward(self, img1, img2=None):
        if self.inferenceMode:
            with torch.no_grad():
                return self.forwardInternal(img1, img2)
        return self.forwardInternal(img1, img2)

    def forwardInternal(self, img1, img2):
        if img2 is not None:
            image_features1 = self.model(img1)
            image_features2 = self.model(img2)
            return image_features1, image_features2
        else:
            image_features = self.model(img1)
            return image_features

    def load(self, modelFile):
        modelState = torch.load(modelFile, weights_only=True)
        missingKeys, unexpectedKeys = self.load_state_dict(modelState, strict=False)
        if len(unexpectedKeys) > 0:
            log.yellow("Unexpected keys found in model file, ignored:\nunexpected={}\nurl={}".format(unexpectedKeys, modelFile))
        if len(missingKeys) > 0:
            log.red("Missing keys in model file:\nmissing={}\nurl={}".format(missingKeys, modelFile))
            import pdb; pdb.set_trace()
        else:
            log.grey("Timm model loaded from file: {}".format(modelFile))

    def save(self, modelFile, verbose=False):
        torch.save(self.state_dict(), modelFile)
        if verbose:
            log.inf("Timm model state saved at {}".format(modelFile))
