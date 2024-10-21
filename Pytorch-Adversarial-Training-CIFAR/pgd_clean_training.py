import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
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
file_name = 'pgd_clean_training'

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

def pgd_linf_targeted(model, x_natural, y_targ, epsilon, alpha, k, device):
    x = x_natural.detach()
    x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    for i in range(k):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            targeted_labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device).fill_(y_targ[0])
            loss = F.cross_entropy(logits[:, y_targ], targeted_labels)
        
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + alpha * torch.sign(grad.detach())
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
    Accuracy =  0
    train_loss = 0
    benign_Acc = 0
    targ_Acc = 0
    benign_train_loss = 0
    tar_train_loss = 0
    tar_correct = 0
    benign_correct = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)



        # clean training
        optimizer.zero_grad()
        output = net(inputs)
        benign_loss = criterion(output, targets)
        benign_loss.backward()
        optimizer.step()



        # Loss and prediction of  Model
        train_loss += benign_loss.item()
        benign_train_loss += benign_loss.item() 
        _, benign_pred = output.max(1)   

        total += targets.size(0)
        benign_correct += benign_pred.eq(targets).sum().item() #The predicted.eq(targets) part compares the predicted values with the target values element-wise

        ben_predicted_label.extend(benign_pred.cpu().numpy())
        
        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign train accuracy:', str(benign_pred.eq(targets).sum().item() / targets.size(0)))
            print('Benign train loss:',benign_loss.item())
    
    benign_Acc = 100. * benign_correct/total

    print(f"\nTotal Benig train accuracy: {np.round(np.array(benign_Acc), decimals=3)}%")

    # Create a DataFrame for the training metrics
    
    train_metrics = train_metrics.append({'Epochs': (epoch+1), 'Train Accuracy': benign_Acc }, ignore_index = True)

    # round off the values
    train_metrics['Train Accuracy'] = train_metrics['Train Accuracy'].round(3)
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')

    return train_metrics

def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    targ_loss = 0
    benign_correct = 0
    adv_correct = 0
    targ_correct = 0
    total = 0
    clean_pred = []
    adv_predicted_list = []
    UnTarg_predicted = []
    targ_predicted_list = []
    True_label = []
    targeted_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            # Evaluate on clean data
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, clean_predicted = outputs.max(1)
            benign_correct += clean_predicted.eq(targets).sum().item()
            clean_pred.extend(clean_predicted.cpu().numpy())

            # Evaluate on untargeted adversarial examples
            x_adv = adversary.perturb(inputs, targets)
            adv_outputs = net(x_adv)
            Untarg_adv_loss = criterion(adv_outputs, targets)
            adv_loss += Untarg_adv_loss.item()

            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(targets).sum().item()
            UnTarg_predicted.extend(adv_predicted.cpu().numpy())


            # Generate random target classes for targeted attack
            # Evaluate on targeted adversarial examples
            y_targ = torch.randint(0, len(class_names), targets.shape, device=device)
            while torch.any(y_targ.eq(targets)):
                y_targ = torch.randint(0, len(class_names), targets.shape, device=device)

            # PGD attack on inputs to generate adversarial examples
            x_adv_targ = pgd_linf_targeted(net, inputs, y_targ, epsilon, alpha, k, device=device)

            targ_outputs = net(x_adv_targ)
            targ_loss += criterion(targ_outputs, y_targ).item()

            _, targ_predicted = targ_outputs.max(1)
            targ_correct += targ_predicted.eq(y_targ).sum().item()
            targ_predicted_list.extend(targ_predicted.cpu().numpy())



            # Transfering data on cpu to plot confusion matrix
            True_label.extend(targets.cpu().numpy())
            targeted_labels.extend(y_targ.cpu().numpy())

    
    benign_Test_Accuracy = 100. * benign_correct / total
    Adv_Test_Aaccuracy = 100. * adv_correct / total
    Targ_Test_Accuracy = 100. * targ_correct / total

    print('\nTotal benign test accuracy:', benign_Test_Accuracy)
    print('Total adversarial test accuracy:', Adv_Test_Aaccuracy)
    print('Total targeted adversarial test accuracy:', Targ_Test_Accuracy)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)
    print('Total targeted adversarial test loss:', targ_loss)

    # Confusion matrices
    clean_conf_matrix = confusion_matrix(True_label, clean_pred)
    Untargeted_conf_matrix = confusion_matrix(True_label, UnTarg_predicted)
    Targeted_conf_matrix = confusion_matrix(True_label, targ_predicted_list)

    # Plot confusion matrices
    plot_confusion_matrix(clean_conf_matrix, class_names, title='Confusion Matrix for Clean Training and Clean Predictions')
    plot_confusion_matrix(Untargeted_conf_matrix, class_names, title='Confusion Matrix for Clean Training and Untargeted Predictions')
    plot_confusion_matrix(Targeted_conf_matrix, class_names, title='Confusion Matrix for Clean Training and Targeted Predictions')




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
    test(epoch)


result_df = pd.concat([train_metrics, test_metrics], axis=1)

result_df.to_csv('Untarg.csv', sep='\t', index=False)
