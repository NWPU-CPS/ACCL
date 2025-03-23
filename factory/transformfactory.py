from misc.log import log
from PIL import Image
import torchvision.transforms as transforms


class JupiterImageTransformFactory(object):
    @classmethod
    def getTransformSimple(cls, imageSize):
        return transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
        ])

    @classmethod
    def getTransformSimpleNorm(cls, imageSize):
        return transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    @classmethod
    def getTransformRotationSimple(cls, imageSize):
        return transforms.Compose([
            transforms.RandomRotation(30, Image.BILINEAR),
            transforms.Resize(imageSize),
            transforms.ToTensor(),
        ])

    @classmethod
    def getTransformColor(cls, imageSize):
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.Resize(imageSize),
            transforms.ToTensor(),
        ])

    @classmethod
    def getTransformFull(cls, imageSize):
        return transforms.Compose([
            transforms.RandomRotation(30, Image.BILINEAR),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3, [0.1, 2.0])], p=0.3),
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    @classmethod
    def getTransformFullNoRotation(cls, imageSize):
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3, [0.1, 2.0])], p=0.3),
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    @classmethod
    def getTransformBrightness(cls, imageSize):
        return transforms.Compose([
            transforms.ColorJitter(brightness=[0.99, 3], contrast=0.5),
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


    @classmethod
    def getTrainTransformSet(cls, name, imageSize):
        log.inf("Image train-transform set initialized: {}/{}".format(name, imageSize))
        if name == "simple":
            return cls.getTransformSimple(imageSize)
        if name == "rotsimple":
            return cls.getTransformRotationSimple(imageSize)
        if name == "color":
            return cls.getTransformColor(imageSize)
        if name == "full":
            return cls.getTransformFull(imageSize)
        if name == "fullnorot":
            return cls.getTransformFullNoRotation(imageSize)
        if name == "brightness":
            return cls.getTransformBrightness(imageSize)
        raise ValueError("Transform not registered: {}".format(name))

    @classmethod
    def getEvalTransformSet(cls, name, imageSize):
        log.inf("Image eval-transform set initialized: {}/{}".format(name, imageSize))
        if name == "simple":
            return cls.getTransformSimple(imageSize)
        if name == "rotsimple":
            return cls.getTransformSimple(imageSize)
        if name == "color":
            return cls.getTransformSimple(imageSize)
        if name == "full":
            return cls.getTransformSimpleNorm(imageSize)
        if name == "fullnorot":
            return cls.getTransformSimpleNorm(imageSize)
        if name == "brightness":
            return cls.getTransformSimpleNorm(imageSize)
        raise ValueError("Transform not registered: {}".format(name))
