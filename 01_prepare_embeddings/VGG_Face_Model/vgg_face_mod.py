# -*- coding: utf-8 -*-

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import torchvision
import torchvision.io as io

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def load_weights(self, path="pretrained/VGG_FACE.t7"):
        """ Function to load luatorch pretrained

        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = self.relu(self.conv_1_1(x))
        x = self.relu(self.conv_1_2(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv_2_1(x))
        x = self.relu(self.conv_2_2(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv_3_1(x))
        x = self.relu(self.conv_3_2(x))
        x = self.relu(self.conv_3_3(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv_4_1(x))
        x = self.relu(self.conv_4_2(x))
        x = self.relu(self.conv_4_3(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv_5_1(x))
        x = self.relu(self.conv_5_2(x))
        x = self.relu(self.conv_5_3(x))
        x = self.max_pool2d(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        return self.fc8(x)

    def get_embedding(
            self,
            x,
            layer="fc6"     # After "fc6", "fc7" or "fc8"
            ):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: Embeddings from fc6 layer

        """
        x = self.relu(self.conv_1_1(x))
        x = self.relu(self.conv_1_2(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv_2_1(x))
        x = self.relu(self.conv_2_2(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv_3_1(x))
        x = self.relu(self.conv_3_2(x))
        x = self.relu(self.conv_3_3(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv_4_1(x))
        x = self.relu(self.conv_4_2(x))
        x = self.relu(self.conv_4_3(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv_5_1(x))
        x = self.relu(self.conv_5_2(x))
        x = self.relu(self.conv_5_3(x))
        x = self.max_pool2d(x)
        x = x.view(x.size(0), -1)
        if layer == "fc6":
            return self.fc6(x)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        if layer == "fc7":
            return self.fc7(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        return self.fc8(x)


if __name__ == "__main__":
    model = VGG_16().double()
    model.load_weights()
    # im = cv2.imread("images/ak.png")
    # im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224).double()
    img_tensor = io.read_image("images/ak.png")  # load image shape: [3, H, W], dtype: uint8 (RGB)
    img_tensor = img_tensor[[2,1,0],:,:].unsqueeze(0).double() # RGB to BGR
    # img_tensor = transform(img_tensor).unsqueeze(0).float() # shape: [1, 3, 224, 224]
    import numpy as np

    model.eval()
    print(img_tensor.shape)
    img_tensor -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
    print(img_tensor.shape)
    # preds = F.softmax(model(img_tensor), dim=1)
    preds = F.softmax(model.get_embedding(img_tensor, layer="fc8"), dim=1)
    values, indices = preds.max(-1)
    print(indices)
