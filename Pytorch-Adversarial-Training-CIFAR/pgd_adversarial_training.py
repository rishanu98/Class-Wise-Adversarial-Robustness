import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import torchvision
import torchvision.transforms as transforms

import os

from models import *

learning_rate = 0.1
epsilon = 0.0314  # Magnitude of the perturbation
k = 7
alpha = 0.00784   # Step size for each perturbation
file_name = 'pgd_adversarial_training'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
True_label = []
UnTarg_predicted = []
targ_clean_pred = []

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
class_names = train_dataset.classes

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x
    

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

def pgd_linf_targeted(model, x_natural, y_targ, epsilon, alpha, k, device):
    x = x_natural.detach()
    x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    for i in range(k):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            targeted_labels = y_targ
            loss = F.cross_entropy(logits, targeted_labels)
        
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() - alpha * torch.sign(grad.detach())  # Note the '-' for maximizing loss
        x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
        x = torch.clamp(x, 0, 1)
    
    return x

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

adversary = LinfPGDAttack(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)   # giving perturbated input in the network
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial train loss:', loss.item())

    print('\nTotal clean train accuarcy:', 100. * correct / total)
    print('Total clean train loss:', train_loss)

def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            y_targ = torch.randint(0, len(class_names), targets.shape).to(device)


            # PGD attack on inputs to generate adversarial examples
            x_adv_targ = pgd_linf_targeted(net, inputs, y_targ, epsilon, alpha, k, device=device)
            adv_outputs = net(x_adv_targ)  

            #outputs = net(inputs)
            loss = criterion(adv_outputs, targets)
            benign_loss += loss.item()

            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(targets).sum().item()
            targ_clean_pred.extend(adv_predicted.cpu().numpy())

            '''if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current benign test accuracy:', str(clean_predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current benign test loss:', loss.item())'''

            '''x = adversary.perturb(inputs, targets)
            adv_outputs = net(x)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()

            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(targets).sum().item()
            UnTarg_predicted.extend(adv_predicted.cpu().numpy())'''

            if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current targeted adversarial test accuracy:', str(adv_predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current targeted adversarial test loss:', loss.item())

            # Transfering data on cpu to plot confusion matrix
            True_label.extend(targets.cpu().numpy())
    
    benign_Test_Accuracy = 100. * benign_correct / total
    Adv_Test_Aaccuracy = 100. * adv_correct / total


    print('\nTotal benign test accuarcy:', benign_Test_Accuracy)
    print('Total adversarial test Accuarcy:', Adv_Test_Aaccuracy)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

# Confusion matrices
    #clean_conf_matrix = confusion_matrix(True_label, clean_pred)
    targeted_conf_matrix = confusion_matrix(True_label, targ_clean_pred)
    
    # Plot confusion matrices
    #plot_confusion_matrix(clean_conf_matrix, class_names, title='Confusion Matrix for Untargeted Training and Clean Predictions')
    plot_confusion_matrix(targeted_conf_matrix, class_names, title='Confusion Matrix for Clean Training and Targeted Predictions')
    
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')
    return benign_Test_Accuracy, Adv_Test_Aaccuracy

def plot_confusion_matrix(conf_matrix, class_names, title):
    df_cm = pd.DataFrame(conf_matrix / np.sum(conf_matrix, axis=1)[:, None], index=class_names, columns=class_names)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(title)
    plt.show()

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(0, 200):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)
