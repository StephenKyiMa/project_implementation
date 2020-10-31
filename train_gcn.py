import torch
import os
import datetime
from torch.utils.data import dataset, dataloader
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
# from torchvision import datasets, transforms
from gcn_model import resnet_gcn
from datasets import *

# ---------- setting -----------#
BATCH_SIZE = 48
N_EPOCHS = 2
CUDA_USE = False
log_interval = 1
WEIGHTS_SAVE_EPOCHS = 5
torch.manual_seed(1)
if CUDA_USE:
    torch.cuda.manual_seed(1)


# ---------- setting -----------#
def main():
    loss_train = 0
    acc_train = 0
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    datatxt = []
    path = './data/train'
    train_loader = DataLoader(
        dataset=PathWiseDataset(path=path, stride=256, rotate=False, flip=False, enhance=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    if CUDA_USE:
        model = resnet_gcn().cuda()
    else:
        model = resnet_gcn()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        scheduler.step()
        stime = datetime.datetime.now()
        correct = 0
        total = 0

        for index, (images, labels) in enumerate(train_loader):
            if CUDA_USE:
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model(Variable(images))
            loss = F.nll_loss(output, Variable(labels))
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(predicted == labels)
            total += len(images)

            if index > 0 and index % log_interval == 0:
                print('Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                    epoch,
                    N_EPOCHS,
                    index * len(images),
                    len(train_loader.dataset),
                    100. * index / len(train_loader),
                    loss.item(),
                    100 * correct / total
                ))

            if int(100. * index / len(train_loader)) >= 98:
                loss_train = float(loss.item())
                acc_train = float(100 * correct / total)

        cur_loss = str(loss_train)
        cur_acc = str(acc_train)
        cur_ep = str(epoch)
        datatxt.append(cur_ep + '_epoch_' + cur_loss + '_loss_' + cur_acc + '_acc')
        with open('train_data.txt', 'w') as f:
            f.write(str(datatxt))

        print('\nEnd of epoch {}, time: {}'.format(epoch, datetime.datetime.now() - stime))

        weights_path = './checkpoints/weights_' + cur_ep + '_epoch_' + cur_loss + '_loss_' + cur_acc + '_acc' + '.pth'
        if epoch % WEIGHTS_SAVE_EPOCHS == 0:
            print('Saving model to "{}"'.format(weights_path))
            torch.save(model.state_dict(), weights_path)


if __name__ == '__main__':
    main()
