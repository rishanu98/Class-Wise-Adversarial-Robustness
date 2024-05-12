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
alp =1e-2
file_name = 'pgd_adversarial_training'
target_indices = [3, 7, 9]

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
class_name = train_dataset.classes

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

class LinfPGDAttack(object):
    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes

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
    
    def pgd_linf_targ2(self, x_natural, y, y_targ, epis, alp, k):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epis, epis)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                targeted_labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device).fill_(y_targ[0])
                loss = F.cross_entropy(logits[:, y_targ], targeted_labels)
       
            grad = torch.autograd.grad(loss, [x])[0]
          
            x = x.detach() + alp * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epis), x_natural + epis)
            x = torch.clamp(x, 0, 1)

        return x
    
    def mixup_data(x, y, alpha = 0.2, device='cuda'):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha).cpu()
        else:
            lam = np.random.beta(alpha, alpha).cpu()

        batch_size = x.size()[0]

        if device == 'cuda':
            index = torch.randperm(batch_size).cuda() # generates a random permutation of indices for shuffling the samples. This is used to select a random sample from the dataset to mix with the current sample.
        else:
            index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        mixed_y = lam * y_a + (1 - lam) * y_b
        return mixed_x, mixed_y
        
    

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv


def choose_target_class(original_class, num_classes):
    # Generate a tensor with possible target classes excluding the original class
    possible_target_classes = torch.tensor([i for i in range(num_classes) if i != original_class])
    
    # Randomly choose a target class from the tensor
    target_class = torch.randint(0, len(possible_target_classes), (1,))
    
    return possible_target_classes[target_class].item()

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
num_classes=10
cudnn.benchmark = True

adversary = LinfPGDAttack(net, num_classes)
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

        #target_classes = [i for i in range(10) if i != targets.item()]

        target_classes = torch.randint(3,7,(targets.size(0),)).to(device)

        assert target_classes.shape == targets.shape, "Shapes do not match"

        if torch.equal(targets, target_classes):
        
            mixed_inputs, mixed_labels = adversary.mixup_data(inputs, targets)
        
        else :
            mixed_inputs = inputs
            mixed_labels = targets


        # Non-adversarial training
        '''optimizer.zero_grad()
        output = net(inputs)
        benign_loss = criterion(output, targets)
        benign_loss.backward()
        optimizer.step()'''

        #Targeted attack
        optimizer.zero_grad()
        adv = adversary.pgd_linf_targ2(mixed_inputs, mixed_labels, torch.randint(0,10,(targets.size(0),)).to(device), epis=0.2, alp= 1e-2, k=7)
        #print("Delta train grad_fn:", adv.data)
        tar_outputs = net(adv)
        targ_loss = criterion(tar_outputs, targets)
        targ_loss.backward()
        optimizer.step()

        # Loss and prediction of Adv_Model and Model
        #train_loss += loss.item()
        #benign_train_loss += benign_loss.item() 
        tar_train_loss += targ_loss.item()
       # _, predicted = adv_outputs.max(1) # adv pred
        #_, benign_pred = output.max(1)   # benign pred
        _, tar_pred = tar_outputs.max(1) #targ pred

        total += targets.size(0)
        #correct += predicted.eq(targets).sum().item() #The predicted.eq(targets) part compares the predicted values with the target values element-wise
       # benign_correct += benign_pred.eq(targets).sum().item()
        tar_correct += tar_pred.eq(targets).sum().item()

        
        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
           # print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
           # print('Current benign train accuracy:', str(benign_pred.eq(targets).sum().item() / targets.size(0)))
            print('Current targeted train accuracy:', str(tar_pred.eq(targets).sum().item() / targets.size(0)))
           # print('Current adversarial train loss:', loss.item())
            #print('Benign train loss:',benign_loss.item())
            print('target train loss:',targ_loss.item())
    
   # Accuracy =  100. * correct / total
    #benign_Acc = 100. * benign_correct/total
    targ_Acc = 100. * tar_correct/total

   # print(f"\nTotal Benig train accuracy: {np.round(np.array(benign_Acc), decimals=3)}%")
    #print(f"Total Benig train Loss: {np.round(np.array(benign_train_loss), decimals=3)}")
   # print(f"Total Adversarial train accuarcy: {np.round(np.array(Accuracy), decimals=3)}%")
    print(f"Total target train accuarcy: {np.round(np.array(targ_Acc), decimals=3)}%")
    


    # Create a DataFrame for the training metrics
    
    #train_metrics = train_metrics.append({'Epochs': (epoch+1), 'Train Accuracy': benign_Acc, 'Targeted Train Accuracy': targ_Acc }, ignore_index = True)
    train_metrics = train_metrics.append({'Epochs': (epoch+1), 'Targeted Train Accuracy': targ_Acc }, ignore_index = True)

    # round off the values 
    train_metrics['Targeted Train Accuracy'] = train_metrics['Targeted Train Accuracy'].round(3)


    return train_metrics

def test(epoch, test_metrics):
    adversarial_test_accuracy = []
    benign_test_accuracy = []
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    test_accuracy = 0
    benign_loss = 0
    ben_test_accuracy = 0
    adv_loss = 0
    Targ_loss=0
    Targeted_correct=0
    targ_test_Accuracy = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            
            # Benign Accuracy
            '''benign_output = net(inputs)
            Benign_loss = criterion(benign_output, targets)
            benign_loss += Benign_loss.item()

            _, benign_predicted = benign_output.max(1)
            benign_correct += benign_predicted.eq(targets).sum().item()'''
            
            #Targeted test Accuracy
            
            delta_Test = adversary.pgd_linf_targ2(inputs, targets, torch.randint(0, 10, (targets.size(0),)).to(device), epis=0.2, alp= 1e-2, k = 7)
            targ_outputs = net(delta_Test)
            targeted_loss = criterion(targ_outputs, targets)
            Targ_loss += targeted_loss.item()

            _, Targeted_predicted = targ_outputs.max(1)
            Targeted_correct += Targeted_predicted.eq(targets).sum().item()

            # Transfering data on cpu to plot confusion matrix
            True_label.extend(targets.cpu().numpy())
           # Adv_predicted_label.extend(predicted.cpu().numpy())
            #ben_predicted_label.extend(benign_predicted.cpu().numpy())
            Target_predicted.extend(Targeted_predicted.cpu().numpy())


            if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                #print('Current benign test accuracy:', str(benign_predicted.eq(targets).sum().item() / targets.size(0)))
                #print('Current benign test loss:', Benign_loss.item())
               # print('Current adversarial test accuracy', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current Targeted test accuracy', str(Targeted_predicted.eq(targets).sum().item() / targets.size(0)))
    
   # ben_test_accuracy = 100. * benign_correct / total
    targ_test_Accuracy = 100. * Targeted_correct/ total

    
   # print(f"\nTotal benign test accuarcy: {round(ben_test_accuracy, 2)}%")
   # print(f"Total adversarial test Accuarcy: {round(adv_test_Accuracy,2)}%")
    print(f"Total Targeted test Accuracy: {round(targ_test_Accuracy,2)}%")
    #print(f"Total benign test loss: {round(benign_loss,3)}")
    print(f"Total adversarial test loss: {round(adv_loss,3)}")

    #test_metrics = test_metrics.append({ 'Test Accuracy': ben_test_accuracy, 'Targeted Test Accuracy': targ_test_Accuracy}, ignore_index = True)
    test_metrics = test_metrics.append({ 'Targeted Test Accuracy': targ_test_Accuracy}, ignore_index = True)

    #test_metrics['Test Accuracy'] = test_metrics['Test Accuracy'].round(3)
    test_metrics['Targeted Test Accuracy'] = test_metrics['Targeted Test Accuracy'].round(3)

    # Compute confusion matrix
    Targ_conf_matrix = confusion_matrix(True_label, Target_predicted)

    # Normalize the confusion matrices conf_matrix / np.sum(conf_matrix, axis=1)[:, None]
    #df_cm = pd.DataFrame(conf_matrix / np.sum(conf_matrix, axis=1)[:, None], index = [i for i in class_name],
     #                columns = [i for i in class_name])
  #  adv_df_cm = pd.DataFrame(Adv_conf_matrix / np.sum(conf_matrix, axis=1)[:, None], index = [i for i in class_name],
                     #columns = [i for i in class_name])
    targ_df_cm = pd.DataFrame(Targ_conf_matrix / np.sum(Targ_conf_matrix, axis=1)[:, None], index = [i for i in class_name],
                     columns = [i for i in class_name])
    
    #fig, axs = plt.subplots(1,1, figsize = (12,7))

    '''sn.heatmap(df_cm, annot=True, cmap='Blues', cbar=False, ax=axs[0])
    axs[0].set_title('Benign Test Confusion Matrix (Normalized)')
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('True')'''

    # Plot the third confusion matrix
    '''sn.heatmap(targ_df_cm, annot=True, cmap='Blues', cbar=False)
    axs[1].set_title('Targeted Test Confusion Matrix  (Normalized)')
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('True')'''
    plt.figure(figsize=(8, 6))
    sn.heatmap(targ_df_cm, annot=True, cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Targeted Attacks(Mix-Up)')


    plt.tight_layout()

    plt.savefig('Conf_Mat_targ(Mix_Up).png')

    plt.show()


    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')

    return test_metrics

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
    test_metrics = test(epoch, test_metrics)


result_df = pd.concat([train_metrics, test_metrics], axis=1)

result_df.to_csv('metrics.csv', sep='\t', index=False)
