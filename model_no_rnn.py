import torch

import torchvision
import torch.nn.functional as F 
from torch import nn
from config import config

def generate_model():
    class DenseModel(nn.Module):
        def __init__(self, pretrained_model):
            super(DenseModel, self).__init__()
            self.classifier = nn.Linear(pretrained_model.classifier.in_features, config.num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

            self.features = pretrained_model.features
            self.layer1 = pretrained_model.features._modules['denseblock1']
            self.layer2 = pretrained_model.features._modules['denseblock2']
            self.layer3 = pretrained_model.features._modules['denseblock3']
            self.layer4 = pretrained_model.features._modules['denseblock4']

        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.avg_pool2d(out, kernel_size=8).view(features.size(0), -1)
            out = F.sigmoid(self.classifier(out))
            return out

    return DenseModel(torchvision.models.densenet169(pretrained=True))

def get_net():
    #return MyModel(torchvision.models.resnet101(pretrained = True))
    model = torchvision.models.resnet34(pretrained = True)
    #for param in model.parameters():
    #    param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(512,config.num_classes)
    return model
'''
from torchvision import models
from pretrainedmodels.models import bninception
from torch import nn
from torch.nn.init import kaiming_normal
import torch
from cnn_finetune import make_model
from config import config

class CNN(nn.Module):
    ## We use ResNet weights from PyCaffe.
    def __init__(self):
        super(CNN, self).__init__()
        
        # Loading pretrained ResNet as feature extractor

        self.model = make_model('densenet121', num_classes=config.num_classes, pretrained=True)
            
        # self.layer1 = nn.Linear(config.num_classes, embedding_size)

        # for m in self.layer1:
        #     kaiming_normal(m.weight)

        for p in self.model.parameters():
            p.requires_grad = True
        
    def forward(self, x):
        f = self.model(x)
        # out = self.layer1(f)
        return f

class DecoderRNN(nn.Module):
    def __init__(self, num_feats, num_classes, hidden_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.rnn = nn.GRU(input_size=num_feats,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first = True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        
        # Init of last layer
        kaiming_normal(self.classifier.weight)
    

    def forward(self, feats, hidden=None):
        x, hidden = self.rnn(feats.unsqueeze(1), hidden)
        x = x.view(-1, self.hidden_size)
        x = self.classifier(x)
        return x

class BuildModel(nn.Module):
    def __init__(self):
        super(BuildModel, self).__init__()
        self.cnn = CNN().cuda()
        self.rnn = DecoderRNN(config.num_classes, config.num_classes, 64, 10).cuda()
    
    def forward(self, x):
        x = self.cnn(x)
        out = self.rnn(x)
        out *= x
        return out

def get_net():
    """
    if config.channels == 4:
        model = models.resnet50(pretrained=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048,config.num_classes)
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
    elif config.channels == 3:
    """
    # model = bninception(pretrained="imagenet")
    # model.global_pool = nn.AdaptiveAvgPool2d(1)
    # model.conv1_7x7_s2 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    # model.last_linear = nn.Linear(1024,config.num_classes)
    # return model
    model = BuildModel()
    return model

'''
if __name__ == '__main__':
    model = get_net()
    # print(model)
    input = torch.autograd.Variable(torch.randn(1, 3, 256, 256))
    # print(input.shape)
    o = model(input)
    # print(o.size())
    from torchsummary import summary

    summary(model, (3, 256, 256))