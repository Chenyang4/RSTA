from torch import nn
import torchvision

from . import resnet
from . import pyramidnet
from . import wide_resnet
from .shakeshake import shake_resnet


class Model(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, num_classes=10, arch=None):
        super(Model, self).__init__()

        resnet_arch = getattr(torchvision.models.resnet, arch)
        net = resnet_arch(num_classes=num_classes)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.feature = nn.Sequential(*self.net[:-1])
        self.classifier = self.net[-1]

    def forward(self, x, statistics=False):
        feat = self.feature(x)  # NxD
        logits = self.classifier(feat)  # NxC
        if statistics:
            return logits, feat
        else:
            return logits


def build_model(model_name, num_classes=10):
    if model_name == 'resnet18':
        model = Model(arch='resnet18', num_classes=num_classes)

    elif model_name in ['wideresnet-28-10', 'wrn-28-10']:
        model = wide_resnet.WideResNet(28, 10, 0, num_classes)

    elif model_name in ['wideresnet-40-2', 'wrn-40-2']:
        model = wide_resnet.WideResNet(40, 2, 0, num_classes)

    elif model_name in ['shakeshake26_2x32d', 'ss32']:
        model = shake_resnet.ShakeResNet(26, 32, num_classes)

    elif model_name in ['shakeshake26_2x96d', 'ss96']:
        model = shake_resnet.ShakeResNet(26, 96, num_classes)

    elif model_name in ['shakeshake26_2x112d', 'ss112']:
        model = shake_resnet.ShakeResNet(26, 112, num_classes)

    elif model_name == 'pyramidnet':
        model = pyramidnet.PyramidNet('cifar10', depth=272, alpha=200, num_classes=num_classes, bottleneck=True)

    elif model_name == 'resnet200':
        model = resnet.ResNet('imagenet', 200, num_classes, True)

    elif model_name == 'resnet50':
        model = resnet.ResNet('imagenet', 50, num_classes, True)

    return model
