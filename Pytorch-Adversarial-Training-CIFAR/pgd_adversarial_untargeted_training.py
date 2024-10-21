import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from models import *

learning_rate = 0.1
epsilon = 0.0314  # Magnitude of the perturbation
k = 7
alpha = 0.00784   # Step size for each perturbation
file_name = 'pgd_adversarial_untargeted_training'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)  # this line is part of the process to generate adversarial examples using the Projected Gradient Descent (PGD) method. The goal is to find a perturbation delta that, when added to the original input X, maximizes the loss with respect to the true labels y
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x
    
    '''def pgd_linf_targ2(self, x_natural , y_targ):
        delta = torch.zeros_like(x_natural, requires_grad=True)
        for t in range(k):
            yp = self.model(x_natural + delta)
            loss = 2*yp[:,y_targ].sum() - yp.sum()
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        return delta.detach()'''
    

    
    def mixup_data(x, y, alpha=1.0, device='cuda'):
   # '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if device=='cuda':
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

def pgd_linf_targeted(model, x_natural, y_targ, epsilon, alpha, k, device=device):
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

# Helper function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, class_names, title):
    df_cm = pd.DataFrame(conf_matrix / np.sum(conf_matrix, axis=1)[:, None], index=class_names, columns=class_names)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.show()

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

adversary = LinfPGDAttack(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

Epochs = []
True_label = []
Adv_predicted_label = []
ben_predicted_label = []
adversarial_train_accuracy = []
Train_Acc = []
Target_predicted = []

train_metrics = pd.DataFrame(columns=['Epochs', 'Train Accuracy', 'Adv Train Accuracy'])
test_metrics = pd.DataFrame(columns=['Test Accuracy', 'Adv Test Accuracy'])



def adversarial_train(epoch, train_metrics):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    targ_correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate random target classes for targeted attack
        # Evaluate on targeted adversarial examples
        #y_targ = torch.randint(0, len(class_names), targets.shape).to(device)

        # PGD attack on inputs to generate adversarial examples
        #x_adv_targ = pgd_linf_targeted(net, inputs, y_targ, epsilon, alpha, k, device=device)

        untarg_adv_data = adversary.perturb(inputs, targets)

        # Forward pass and calculate loss
        optimizer.zero_grad()
        outputs = net(untarg_adv_data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Record statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        targ_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}:")
            print('Current targeted train accuracy:', predicted.eq(targets).sum().item() / targets.size(0))
            print('Targeted train loss:', loss.item())

    # Calculate metrics
    train_accuracy = 100. * targ_correct / total
    print(f'Epochs: {epoch} and Train_accuracy: {train_accuracy}')
    # Update train metrics dataframe
    train_metrics = train_metrics.append({'Epochs': epoch, 'Train Accuracy': train_accuracy}, ignore_index=True)
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')
    return train_metrics

def eval_test_new(epoch,model, device, test_loader):
    model.eval()
    clean_correct = 0
    untargeted_correct = 0
    targeted_correct = 0
    total = 0

    true_clean_labels = []
    clean_prediction = []

    true_targeted_labels = []
    targeted_prediction = []

    true_untargeted_labels = []
    untargeted_prediction = []

    with torch.no_grad():

        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            total+=targets.size(0)

            #mix_x, y_a, y_b, lam = mixup_data(data, targets, alpha = 0.2, device='cuda')


            # clean accuracy with mix_up
            clean_output = model(data)
            _ , clean_pred = clean_output.max(1)
            clean_correct += clean_pred.eq(targets).sum().item()

            # save to plot confusion matrix
            clean_prediction.extend(clean_pred.cpu().numpy())
            true_clean_labels.extend(targets.cpu().numpy())

            # targeted accuracy with mix_up

            y_targ = torch.randint(0, len(class_names), targets.shape).to(device) # target label generation for targeted accuracy
            adv_data = pgd_linf_targeted(model, data, y_targ, epsilon=epsilon, alpha=0.01, k=10)
            adv_output = model(adv_data)

            _, predicted_targeted = adv_output.max(1)
            targeted_correct += predicted_targeted.eq(y_targ).sum().item()
            targeted_prediction.extend(predicted_targeted.cpu().numpy())

            #Untargeted accuracy 
            untarg_adv_data = adversary.perturb(data, targets)
            untargeted_adv_output = model(untarg_adv_data)

            _, predicted_untargeted = untargeted_adv_output.max(1)
            untargeted_correct += predicted_untargeted.eq(targets).sum().item()
            untargeted_prediction.extend(predicted_untargeted.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f'Current Epoch: {epoch}')
                print(f'Batch {batch_idx}/{len(test_loader)}:')
                print(f'  Clean Accuracy: {100. * clean_correct / total:.2f}%')
                print(f'  Targeted Attack Accuracy: {100. * targeted_correct / total:.2f}%')
                print(f'  Untargeted Attack Accuracy: {100. * untargeted_correct / total:.2f}%')

    clean_acc = 100. * clean_correct/total
    targeted_acc = 100. * targeted_correct/total
    untargeted_acc = 100. * untargeted_correct/total

    print(f'Final Clean Accuracy: {100. * clean_acc / total:.2f}%')
    print(f'Final Targeted Attack Accuracy: {100. * targeted_acc / total:.2f}%')
    print(f'Final Untargeted Attack Accuracy: {100. * untargeted_acc / total:.2f}%')

    # plot the confusion matrix
    clean_conf_matrix = confusion_matrix(true_clean_labels, clean_prediction)
    plot_confusion_matrix(clean_conf_matrix, class_names, title='Confusion Matrix for PGD untargeted training and clean testing')

    # Plot confusion matrix for targeted attack predictions
    targeted_conf_matrix = confusion_matrix(true_clean_labels, targeted_prediction)
    plot_confusion_matrix(targeted_conf_matrix, class_names, title='Confusion Matrix for PGD untargeted training and targeted testing')

    # Plot confusion matrix for untargeted attack predictions
    untargeted_conf_matrix = confusion_matrix(true_clean_labels, untargeted_prediction)
    plot_confusion_matrix(untargeted_conf_matrix, class_names, title='Confusion Matrix for PGD untargeted training and Untargeted testing')

    return clean_acc, targeted_acc, untargeted_acc



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
    train_metrics = adversarial_train(epoch, train_metrics)
    eval_test_new(epoch, net, device, test_loader)


result_df = pd.concat([train_metrics, test_metrics], axis=1)

result_df.to_csv('Untarg.csv', sep='\t', index=False)
    
