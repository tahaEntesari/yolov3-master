import copy
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.quantization import fuse_modules
from torch.utils.data import Dataset

from models.common import Conv


def fuseAll(model):
    for m in model.modules():
        if type(m) == Conv:
            fuse_modules(m, ['conv', 'bn'], inplace=True)


class CustomDataSet(Dataset):
    def __init__(self, imgSize):
        super(CustomDataSet, self).__init__()
        self.imagenetList = os.listdir("./imagenetImages")
        self.imgSize = imgSize

    def __len__(self):
        return len(self.imagenetList)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imageName = self.imagenetList[idx]
        print(imageName)
        img = cv2.imread("./imagenetImages/" + imageName)
        # Padded resize
        img = letterbox(img, new_shape=self.imgSize)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(torch.device("cpu"))
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return [], img, [], []


class NewModelTinyParent(nn.Module):
    def __init__(self, model, quantizedNewModel):
        super(NewModelTinyParent, self).__init__()
        self.zeroPad = model.model[11]
        self.detect = model.model[20]
        self.quantizedNewModel = quantizedNewModel

    def forward(self, x):
        return self.quantizedNewModel(x, self.zeroPad, self.detect)


class NewModelTiny(nn.Module):
    def __init__(self, model):
        super(NewModelTiny, self).__init__()
        detectLayerIndex = 20
        zeroPadLayerIndex = 11
        self.quant1 = torch.quantization.QuantStub()
        self.quant2 = torch.quantization.QuantStub()

        self.model = copy.deepcopy(model)
        self.model.model[zeroPadLayerIndex] = nn.Identity()
        self.model.model[zeroPadLayerIndex].breakHere = False

        self.model.model[detectLayerIndex] = nn.Identity()
        self.model.model[detectLayerIndex].breakHere = True
        self.model.model[detectLayerIndex].f = model.model[detectLayerIndex].f

        self.deQuant = torch.quantization.DeQuantStub()

    def forward(self, x, zeroPad, detect):
        x = self.quant1(x)
        x = self.model(x, zeroPad=zeroPad, deQuant=self.deQuant, quant=self.quant2)
        for i in range(len(x)):
            x[i] = self.deQuant(x[i])
        x = detect(x)
        return x


# class NewModelTiny(nn.Module):
#     def __init__(self, model):
#         super(NewModelTiny, self).__init__()
#         detectLayerIndex = 20
#         zeroPadLayerIndex = 11
#         self.quant1 = torch.quantization.QuantStub()
#         self.quant2 = torch.quantization.QuantStub()
#         self.model1 = copy.deepcopy(model)
#         self.model2 = copy.deepcopy(model)
#         for i in range(zeroPadLayerIndex, detectLayerIndex + 1):
#             self.model1.model[i] = nn.Identity()
#             self.model1.model[i].breakHere = True
#         for i in range(zeroPadLayerIndex + 1):
#             self.model2.model[i] = nn.Identity()
#             self.model2.model[i].breakHere = False
#         self.model2.model[detectLayerIndex] = nn.Identity()
#         self.model2.model[detectLayerIndex].breakHere = True
#         self.model2.model[detectLayerIndex].f = model.model[detectLayerIndex].f
#         self.deQuant = torch.quantization.DeQuantStub()
#
#     def forward(self, x, zeroPad, detect):
#         print(self.model1)
#         print("********************\n" * 10)
#         print(self.model2)
#         x = self.quant1(x)
#         x = self.model1(x)
#         x = self.deQuant(x)
#         print(x.shape)
#         x = zeroPad(x)
#         print(x.shape)
#         x = self.quant2(x)
#         x = self.model2(x)
#         for i in range(len(x)):
#             x[i] = self.deQuant(x[i])
#         x = detect(x)
#         return x


class NewModel2(nn.Module):
    def __init__(self, model, detectLayerIndex, quantizedNewModel):
        super(NewModel2, self).__init__()
        self.detect = model.model[detectLayerIndex]
        self.newModel = quantizedNewModel

    def forward(self, x):
        x = self.newModel(x)
        x = self.detect(x)
        return x


class NewModel(nn.Module):
    def __init__(self, model, detectLayerIndex=28):
        super(NewModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        # self.model = model
        self.model = copy.deepcopy(model)
        # self.detect = self.model.model[detectLayerIndex]
        self.model.model[detectLayerIndex] = nn.Identity()
        self.model.model[detectLayerIndex].breakHere = True
        self.model.model[detectLayerIndex].f = model.model[detectLayerIndex].f
        self.deQuant = nn.ModuleList([torch.quantization.DeQuantStub(),
                                      torch.quantization.DeQuantStub(),
                                      torch.quantization.DeQuantStub()])

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        for i in range(len(x)):
            x[i] = self.deQuant[i](x[i])
        return x


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
