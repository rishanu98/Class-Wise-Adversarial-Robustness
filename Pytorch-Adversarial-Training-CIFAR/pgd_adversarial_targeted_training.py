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
file_name = 'pgd_adversarial_targeted_training'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
True_label = []
UnTarg_predicted = []
targ_clean_pred = []

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


mu = torch.tensor(cifar10_mean).view(3,1,1)
std = torch.tensor(cifar10_std).view(3,1,1)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    #stransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
class_names = train_dataset.classes

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

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

def normalize_cifar(x):
    return (x - mu.to(x.device))/(std.to(x.device))

def pgd_linf_targeted(model, x_natural, y_targ, epsilon, alpha, k, device):
    x = x_natural.detach()
    x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    for i in range(k):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits[:,y_targ], y_targ)
        
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


def plot_confusion_matrix(confusion_matrix, class_names, title, cmap='Blues'):
    """
    Function to plot a confusion matrix for visualization.

    Parameters:
    - confusion_matrix (ndarray): The confusion matrix to plot.
    - class_names (list): List of class names corresponding to the labels.
    - title (str): Title of the confusion matrix plot.
    - cmap (str): Colormap for the confusion matrix. Default is 'Blues'.
    
    Returns:
    - A matplotlib plot of the confusion matrix.
    """
    plt.figure(figsize=(10, 8))  # Set figure size
    df_cm = pd.DataFrame(confusion_matrix / np.sum(confusion_matrix, axis=1)[:, None], index=class_names, columns=class_names)
    sn.heatmap(df_cm, annot=True, fmt='.3f', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names, square=True)

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()  # Adjust the plot to fit the labels properly
    plt.savefig(f"{title}.png")
    plt.show()

def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        y_targ = torch.randint(0, len(class_names), targets.shape).to(device)  # Simple shifting of target class


        # PGD attack on inputs to generate adversarial examples
        x_adv_targ = pgd_linf_targeted(net, inputs, y_targ, epsilon, alpha, k, device=device)
        adv_outputs = net(x_adv_targ)
        loss = criterion(adv_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = adv_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current adversarial targeted train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial targeted train loss:', loss.item())

    print('\nTotal adversarial targeted train accuarcy:', 100. * correct / total)
    print('Total adversarial targeted train loss:', train_loss)
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')

def test(epoch):
    print(f'Test epoch : {epoch}')
    net.eval()  # Switch the model to evaluation mode
    
    benign_loss = 0
    benign_correct = 0
    total = 0

    all_true_labels = []  # For storing true labels
    all_predicted_labels = []  # For storing predictions
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            # Forward pass
            #y_targ = (targets + 1) % 10  # Simple shifting of target class


        # PGD attack on inputs to generate adversarial examples
            x_adv_targ = adversary.perturb(inputs, targets) #peturbated input
            #adv_outputs = net(normalize_cifar(x_adv_targ))
            outputs = net(x_adv_targ)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            # Predictions
            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            # Save true labels and predictions for confusion matrix
            all_true_labels.extend(targets.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f'\nCurrent batch: {batch_idx}')
                print(f'Current benign test accuracy: {predicted.eq(targets).sum().item() / targets.size(0):.4f}')
                print(f'Current benign test loss: {loss.item():.4f}')
    
    # After the loop, calculate final accuracy
        benign_test_accuracy = 100. * benign_correct / total
        print(f'\nTotal benign test accuracy: {benign_test_accuracy:.2f}%')

    # Generate the confusion matrix
        targeted_conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
    
    # Plot the confusion matrix
        plot_confusion_matrix(targeted_conf_matrix, class_names, title='Confusion Matrix for Targeted Training and Untargeted Predictions')



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
