from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from models.wide_resnet import *
from models.resnet import *
#from models import *
from train_mode.stop_to_lastclean import *
from train_mode.stop_to_firstadv import *
import torchvision.models as models
import random
from utils.attacks import pgd_linf,mixup_data, pgd_linf_targ2

from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./save_model/resnet_uniform_trades_like_fat',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

#cifar10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
class_names = trainset.classes

def BAT_loss(firstadv_logits, lastclean_logits, target, beta, clean_logits):
    batch_size = len(target)
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()
    loss_natural = nn.CrossEntropyLoss(reduction='mean')(lastclean_logits, target)
    uniform_logits = torch.ones_like(firstadv_logits).cuda() * 0.1
    loss_uniform_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(uniform_logits, dim=1),
                                                         F.softmax(lastclean_logits, dim=1))
    loss_uniform_robust2 = (1.0 / batch_size) * criterion_kl(F.log_softmax(uniform_logits, dim=1),
                                                         F.softmax(firstadv_logits, dim=1))
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(firstadv_logits, dim=1),
                                                         F.softmax(clean_logits, dim=1))
    loss_uniform = (loss_uniform_robust + loss_uniform_robust2) 
    return loss_natural, beta * loss_robust, loss_uniform


def train(args, model, device, train_loader, optimizer, epoch, ema=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if epoch > 500 and epoch % 2 == 0:
            print_flag = True
        else:
            print_flag = False

        '''target_classes = torch.randint(3,7,(target.size(0),)).to(device)   # create a target class tensor which only includes hard classes 

        assert target_classes.shape == target.shape, "Shapes do not match"

        if torch.equal(target, target_classes):
        
            mixed_inputs, mixed_labels = mixup_data(data, target, alpha = 0.2, device='cuda')
        else:
            mixed_inputs = data
            mixed_labels = target'''
    
        last_clean, lastclean_target, _, _ = stop_to_lastclean(model, data, target, print_flag, step_size=args.step_size,
                                                                    epsilon=args.epsilon, perturb_steps=args.num_steps,
                                                                    randominit_type="normal_distribution_randominit", loss_fn='kl') 

        first_adv, _, output_natural, _ = stop_to_firstadv(model, data, target, step_size=args.step_size,
                                                                    epsilon=args.epsilon, perturb_steps=args.num_steps,
                                                                    randominit_type="normal_distribution_randominit", loss_fn='kl',tau=1) 

        model.train()
        optimizer.zero_grad()
        lastclean_logits = model(last_clean)
        firstadv_logits = model(first_adv)
        clean_logits = model(output_natural)


        loss_natural, loss_robust, loss_uniform = BAT_loss(firstadv_logits, lastclean_logits, lastclean_target, args.beta, clean_logits)
        loss = loss_robust + loss_natural + loss_uniform

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            with open('Results_1.txt', 'a') as file:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()),file=file)


def eval_train(model, epoch,device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    with open('Results_1.txt', 'a') as file:
        print('Epoch: {}.\n'.format(epoch), file=file)
        print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%).\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)), file=file)
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model,device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    True_label = []
    predicted_label = []
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(device)
            y_targ = torch.randint(0, len(class_names), target.shape).to(device)
            adv_data = pgd_linf_targ2(model, data, y_targ, epis=args.epsilon, alp=0.01, k=10)
            output = model(adv_data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Transfering data on cpu to plot confusion matrix
            True_label.extend(target.cpu().numpy())
            predicted_label.extend(pred.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    with open('Results_3.txt', 'a') as file:
        print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%.\n)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),file=file)
    test_accuracy = correct / len(test_loader.dataset)
    conf_matrix = confusion_matrix(True_label, predicted_label)

    targ_df_cm = pd.DataFrame(conf_matrix / np.sum(conf_matrix, axis=1)[:, None], index = [i for i in class_names],
                     columns = [i for i in class_names])
    
    plt.figure(figsize=(8, 6))
    sn.heatmap(targ_df_cm, annot=True, cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for targeted Attacks and targeted testing')


    plt.tight_layout()

    plt.savefig('./RESULTS/Confusion_Matrix_for_targeted Attacks_and_targeted_testing.png')

    plt.show()

    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    model = ResNet18()
    model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        with open('Results_1.txt','a') as file:
            if epoch == 1:
                print('Results for Targeted Training and Untargeted Testing',file=file)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        with open('Results_1.txt','a') as file:
            print('================================================================',file=file)
            eval_train(model, epoch,device, train_loader)
            eval_test(model,device, test_loader)
            print('================================================================',file=file)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
