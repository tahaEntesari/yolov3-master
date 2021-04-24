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


class NewModel2(nn.Module):
    def __init__(self, model, detectLayerIndex, quantizedNewModel):
        super(NewModel2, self).__init__()
        self.detect = model.model[detectLayerIndex]
        self.newModel = quantizedNewModel

    def forward(self, x):
        return self.detect(self.newModel(x))


class NewModel(nn.Module):
    def __init__(self, model, detectLayerIndex):
        super(NewModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        # self.model = model
        self.model = copy.deepcopy(model)
        # self.detect = self.model.model[detectLayerIndex]
        self.model.model[detectLayerIndex] = nn.Identity()
        self.model.model[detectLayerIndex].FAF = True
        self.model.model[detectLayerIndex].f = model.model[detectLayerIndex].f
        if detectLayerIndex == 20:
            self.deQuant = nn.ModuleList([torch.quantization.DeQuantStub(),
                                          torch.quantization.DeQuantStub()])
        elif detectLayerIndex == 28:
            self.deQuant = nn.ModuleList([torch.quantization.DeQuantStub(),
                                          torch.quantization.DeQuantStub(),
                                          torch.quantization.DeQuantStub()])

    def forward(self, x):
        temp = self.model(self.quant(x))
        features = []
        for i in range(len(temp)):
            features.append(self.deQuant[i](temp[i]))
        # features = [dequant(y) for dequant, y in zip(self.deQuant, temp)]
        # return self.detect(features)
        return features


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')
