import torch
import torch.nn as nn
from misc.log import log


class VitModel(nn.Module):
    @classmethod
    def createTrainModel(cls, mcfg):
        model = VitModel(mcfg)
        backboneUrl = mcfg.backboneUrl()
        if backboneUrl is not None:
            pretrainedState = torch.hub.load_state_dict_from_url(
                url=backboneUrl,
                map_location="cpu",
                progress=False,
                model_dir=mcfg.cacheDir(),
            )
            missingKeys, unexpectedKeys = model.backbone.load_state_dict(pretrainedState, strict=False)
            if len(unexpectedKeys) > 0:
                log.yellow("Unexpected keys found in model url, ignored:\nunexpected={}\nurl={}".format(unexpectedKeys, backboneUrl))
            if len(missingKeys) > 0:
                log.red("Missing keys in model url:\nmissing={}\nurl={}".format(missingKeys, backboneUrl))
                import pdb; pdb.set_trace()
            else:
                log.grey("Pretrained backbone weights loaded from url: {}".format(backboneUrl))
        model.setInferenceMode(False)
        model.train()
        return model

    @classmethod
    def createEvalModel(cls, mcfg, modelFile):
        model = VitModel(mcfg)
        model.load(modelFile)
        model.setInferenceMode(True)
        model.eval()
        return model

    @classmethod
    def initBackbone(cls, mcfg):
        if mcfg.phase == "vitbase8":
            from .components.vit import VisionTransformer
            return VisionTransformer(patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True)
        if mcfg.phase == "vitbase16":
            from .components.vit import VisionTransformer
            return VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True)
        if mcfg.phase == "vitsmall8":
            from .components.vit import VisionTransformer
            return VisionTransformer(patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)
        if mcfg.phase == "vitsmall16":
            from .components.vit import VisionTransformer
            return VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)
        raise ValueError("Invalid model phase for backbone: {}".format(mcfg.phase))

    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.backbone = self.initBackbone(mcfg)
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.embed_dim),
            nn.Linear(self.backbone.embed_dim, mcfg.outDim)
        )
        self.inferenceMode = False
        self.backboneFreezed = False

    def setFreezeBackbone(self, flag):
        self.backboneFreezed = flag
        requiresGrad = not flag
        for name, m in self.backbone.named_parameters():
            if "block" in name:
                blockIndex = int(name.split(".")[1])
                if blockIndex < self.mcfg.numBlockFreezed:
                    m.requires_grad = requiresGrad

    def forwardInternal(self, x):
        x = self.backbone(x)
        return self.head(x)

    def forward(self, x):
        if self.inferenceMode:
            with torch.no_grad():
                return self.forwardInternal(x)
        return self.forwardInternal(x)

    def load(self, modelFile):
        modelState = torch.load(modelFile, weights_only=True)
        missingKeys, unexpectedKeys = self.load_state_dict(modelState, strict=False)
        if len(unexpectedKeys) > 0:
            log.yellow("Unexpected keys found in model file, ignored:\nunexpected={}\nurl={}".format(unexpectedKeys, modelFile))
        if len(missingKeys) > 0:
            log.red("Missing keys in model file:\nmissing={}\nurl={}".format(missingKeys, modelFile))
            import pdb; pdb.set_trace()
        else:
            log.grey("Vit model loaded from file: {}".format(modelFile))

    def save(self, modelFile, verbose=False):
        torch.save(self.state_dict(), modelFile)
        if verbose:
            log.inf("Vit model state saved at {}".format(modelFile))

    def setInferenceMode(self, flag):
        self.inferenceMode = flag
