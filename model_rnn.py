import torchvision
import torch.nn.functional as F 
from config import config
from torchvision import models
from pretrainedmodels.models import bninception
from torch import nn
import torch
from cnn_finetune import make_model
from config import config

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Loading pretrained ResNet as feature extractor

        self.model = make_model('resnet34', num_classes=config.num_classes, pretrained=True)
            
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
        # self.cnn = CNN().cuda()
        self.cnn = CNN()
        # self.rnn = DecoderRNN(config.num_classes, config.num_classes, 64, 10).cuda()
        self.rnn = DecoderRNN(config.num_classes, config.num_classes, 64, 10)
    
    def forward(self, x):
        x = self.cnn(x)
        # out = self.rnn(x)
        # out *= x
        return x

def get_net():
    """
    if config.channels == 4:
        model = models.resnet34(pretrained=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048,config.num_classes)
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
    elif config.channels == 3:
    """
    model = BuildModel()
    return model

if __name__ == '__main__':
    model = get_net()
    # print(model)
    input = torch.autograd.Variable(torch.randn(1, 3, 256, 256))
    # print(input.shape)
    o = model(input)
    # print(o.size())
    from torchsummary import summary
    summary(model, (3, 256, 256))
