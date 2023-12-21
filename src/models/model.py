from typing import Optional

from torchvision.models import resnet18, resnet50
import torchvision.models as models

from torch.nn import Linear
from torchvision.models._api import Weights


def get_model(name: str, weights: Optional[Weights]):
    if name == 'resnet18':
        model = resnet18(weights=weights)
        model.fc = Linear(in_features=512, out_features=45, bias=True)
    elif name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.fc = Linear(in_features=256, out_features=45, bias=True)
    elif name == 'resnet50':
        model = resnet18(weights=weights)
        model.fc = Linear(in_features=512, out_features=45, bias=True)
    else:
        raise ValueError(f'Unsupported model "{name}"')

    return model
