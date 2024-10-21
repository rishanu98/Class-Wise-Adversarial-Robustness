import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import seaborn as sn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from resnet import ResNet18




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

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


mu = torch.tensor(cifar10_mean).view(3,1,1)
std = torch.tensor(cifar10_std).view(3,1,1)

def normalize_cifar(x):
    return (x - mu.to(x.device))/(std.to(x.device))
        
net = ResNet18()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = torch.device('cpu')
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
adversary = LinfPGDAttack(net)
criterion = nn.CrossEntropyLoss()


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

def eval_test_new(model, device, test_loader, class_names):
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
            clean_output = model(normalize_cifar(data))
            _ , clean_pred = clean_output.max(1)
            clean_correct += clean_pred.eq(targets).sum().item()

            # save to plot confusion matrix
            clean_prediction.extend(clean_pred.cpu().numpy())
            true_clean_labels.extend(targets.cpu().numpy())

            # targeted accuracy with mix_up

            y_targ = torch.randint(0, len(class_names), targets.shape).to(device) # target label generation for targeted accuracy
            adv_data = pgd_linf_targeted(model, data, y_targ, epsilon=epsilon, alpha=0.01, k=10)
            adv_output = model(normalize_cifar(adv_data))

            _, predicted_targeted = adv_output.max(1)
            targeted_correct += predicted_targeted.eq(targets).sum().item()
            targeted_prediction.extend(predicted_targeted.cpu().numpy())

            #Untargeted accuracy 
            untarg_adv_data = adversary.perturb(data, targets)
            untargeted_adv_output = model(normalize_cifar(untarg_adv_data))

            _, predicted_untargeted = untargeted_adv_output.max(1)
            untargeted_correct += predicted_untargeted.eq(targets).sum().item()
            untargeted_prediction.extend(predicted_untargeted.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(test_loader)}:')
                print(f'  Clean Accuracy: {100. * clean_correct / total:.2f}%')
                print(f'  Targeted Attack Accuracy: {100. * targeted_correct / total:.2f}%')
                print(f'  Untargeted Attack Accuracy: {100. * untargeted_correct / total:.2f}%')

    clean_acc = 100. * clean_correct/total
    targeted_acc = 100. * targeted_correct/total
    untargeted_acc = 100. * untargeted_correct/total
    

    print(f'Final Clean Accuracy: {clean_acc}%')
    print(f'Final Targeted Attack Accuracy: {targeted_acc}%')
    print(f'Final Untargeted Attack Accuracy: {untargeted_acc}%')

    # plot the confusion matrix
    clean_conf_matrix = confusion_matrix(true_clean_labels, clean_prediction)
    plot_confusion_matrix(clean_conf_matrix, class_names, title='Confusion Matrix for CFA(N) untargeted training and clean testing')

    # Plot confusion matrix for targeted attack predictions
    targeted_conf_matrix = confusion_matrix(true_clean_labels, targeted_prediction)
    plot_confusion_matrix(targeted_conf_matrix, class_names, title='Confusion Matrix for CFA(N) untargeted training and targeted testing')

    # Plot confusion matrix for untargeted attack predictions
    untargeted_conf_matrix = confusion_matrix(true_clean_labels, untargeted_prediction)
    plot_confusion_matrix(untargeted_conf_matrix, class_names, title='Confusion Matrix for CFA(N) untargeted training and Untargeted testing')

    return clean_acc, targeted_acc, untargeted_acc


if __name__ == '__main__':
    
    


    learning_rate = 0.1
    epsilon = 0.0314  # Magnitude of the perturbation
    k = 7
    alpha = 0.00784   # Step size for each perturbation
    file_name = 'AT_CCM_Normalized'   # Change file name to load respective weights of the model



    checkpoint = torch.load('./CFA/save_model/' + file_name)
    net.module.load_state_dict(checkpoint['net'])
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    #state_dict_path = './Pytorch-Adversarial-Training-CIFAR/checkpoint/pgd_clean_training'

    adversary = LinfPGDAttack(net)
    criterion = nn.CrossEntropyLoss()
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    class_names = test_dataset.classes



    eval_test_new(net,device,test_loader,class_names)