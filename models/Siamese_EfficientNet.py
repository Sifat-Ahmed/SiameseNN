import timm
import torch
import torch.nn as nn

class SiameseEff(nn.Module):
    def __init__(self, model_name='efficientnet_b0', out_features=4, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)


    def forward_once(self, x):
        bs = x.size(0)
        output = self.model(x)
        output = self.pooling(output).view(bs, -1)
        return output

    def forward(self, image_1, image_2):
        output1 = self.forward_once(image_1)
        output2 = self.forward_once(image_2)

        output = torch.sqrt(torch.sum((output1 - output2) * (output1 - output2), 1))

        return output1, output2, output