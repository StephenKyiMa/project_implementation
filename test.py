import torch
import os
import datetime
import ntpath
from torch.utils.data import dataset, DataLoader
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
# from torchvision import datasets, transforms
from model import ResNet
from datasets import *
# ---------- setting -----------#
CUDA_USE = True
test_path = './data/test'
verbose = True
weights_path = './checkpoints/weights.pth'
torch.manual_seed(1)
if CUDA_USE:
    torch.cuda.manual_seed(1)
# ---------- setting -----------#

def main():
    if CUDA_USE:
        cnn_res = ResNet().cuda()
    else:
        cnn_res = ResNet()
    
    if os.path.exists(weights_path):
        print('Loading "patch-wise" model...')
        cnn_res.load_state_dict(torch.load(weights_path))
    else:
        print('Failed to load pre-trained network')
    
    cnn_res.eval()
    dataset_test = TestDataset(path=test_path, stride=256, augment=False)
    data_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)
    stime = datetime.datetime.now()
    
    if verbose:
        print('\t\t test\t\t')
    
    for index, (image, file_name) in enumerate(data_loader):
        image = image.squeeze()
        
        if CUDA_USE:
            image = image.cuda()
        with torch.no_grad():
            output = cnn_res(Variable(image))
        _, predicted = torch.max(output.data, 1)
        
        test_label = 3 - np.argmax(np.sum(np.exp(output.data.cpu().numpy()), axis=0)[::-1])
        
        if verbose:
            print('{}) \t {}  \t {}'.format(
                str(index + 1).rjust(2, '0'),
                LABELS[test_label].ljust(8),
                ntpath.basename(file_name[0])))
    
    if verbose:
        print('\nInference time: {}\n'.format(datetime.datetime.now() - stime))


if __name__ == '__main__':
    main()
