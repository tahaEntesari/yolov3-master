from Utilities import *
from models.common import Bottleneck
from models.experimental import attempt_load
from models.yolo import Model, Detect

backend = 'fbgemm'
if 'qnnpack' in torch.backends.quantized.supported_engines:
    backend = 'qnnpack'
torch.backends.quantized.engine = backend


def main():
    imgSize = 640
    pruned = True
    weightPath = "quantizedPrunedWeightsYoloTiny.pth"
    model = Model("./models/yolov3-tiny.yaml", nc=10)
    if pruned:
        prunedModelLocation = "tiny_pruned_best6.pt"
        detectLayerIndex = 20
        tempModel = attempt_load(prunedModelLocation, map_location=torch.device("cpu"))  # load FP32 model
        for layerNumber in range(detectLayerIndex + 1):
            # replace the necessary ones in that layer
            currentLayerType = type(model.model[layerNumber])
            if currentLayerType == Bottleneck:
                replacePruneModuleBottleneck(model.model[layerNumber], tempModel.model[layerNumber])
            elif currentLayerType == nn.Sequential:
                replacePruneModuleSequential(model.model[layerNumber], tempModel.model[layerNumber])
            elif currentLayerType == Conv:
                model.model[layerNumber].conv = tempModel.model[layerNumber].conv
                model.model[layerNumber].bn = tempModel.model[layerNumber].bn

            if layerNumber != detectLayerIndex:
                # replace the first component of the next layer
                nextLayerType = type(model.model[layerNumber + 1])
                if nextLayerType == Bottleneck:
                    model.model[layerNumber + 1].cv1.conv = tempModel.model[layerNumber + 1].cv1.conv
                elif nextLayerType == Conv:
                    model.model[layerNumber + 1].conv = tempModel.model[layerNumber + 1].conv
                elif nextLayerType == nn.Sequential:
                    model.model[layerNumber + 1][0].cv1.conv = tempModel.model[layerNumber + 1][0].cv1.conv
                elif nextLayerType == Detect:
                    for i in range(len(list(model.model[detectLayerIndex].children())[0])):
                        model.model[layerNumber + 1].m[i] = tempModel.model[layerNumber + 1].m[i]

    model.eval()
    fuseAll(model)

    newModel = NewModelTiny(model)
    newModel.qconfig = torch.quantization.get_default_qconfig(backend)
    newModel.eval()
    mQuan = torch.quantization.prepare(newModel)
    newModelParent = NewModelTinyParent(model, mQuan)
    newModelParent.eval()

    randomInput = torch.rand((1, 3, imgSize, imgSize))
    newModelParent(randomInput)
    mQuan = torch.quantization.convert(mQuan)
    model = NewModelTinyParent(model, mQuan)
    model.quantizedNewModel.load_state_dict(torch.load(weightPath))


if __name__ == '__main__':
    main()
