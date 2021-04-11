import torch
import torch.nn as nn
import os
from models.common import Conv
from torch.quantization import fuse_modules
import copy


def fuseAll(model):
    for m in model.modules():
        if type(m) == Conv:
            fuse_modules(m, ['conv', 'bn'], inplace=True)


class NewModel(nn.Module):
    def __init__(self, model, detectLayerIndex):
        super(NewModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        # self.model = model
        self.model = copy.deepcopy(model)
        self.detect = self.model.model[detectLayerIndex]
        self.model.model[detectLayerIndex] = nn.Identity()
        self.model.model[detectLayerIndex].FAF = True
        self.model.model[detectLayerIndex].f = self.detect.f

    def forward(self, x):
        features = [self.dequant(y) for y in self.model(self.quant(x))]
        return self.detect(features)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')
