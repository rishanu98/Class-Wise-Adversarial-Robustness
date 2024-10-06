import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

alpha = 0.00784
criterion = nn.CrossEntropyLoss()
def pgd_attack(model,
                  X,
                  y,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size):

    device = X.device
    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True

    for i in range(num_steps):

        pred = model(X_pgd)
        loss = nn.CrossEntropyLoss()(pred, y)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    return X_pgd

def pgd_linf(model, X, y, epsilon, alpha, num_iter, randomize=False):
    """ Construct PGD adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf_targeted(model, x_natural, y_targ, epsilon, alpha, k, device):
    x = x_natural.detach()
    x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    for i in range(k):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            targeted_labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device).fill_(y_targ[0])
            loss = F.cross_entropy(logits, targeted_labels)
        
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + alpha * torch.sign(grad.detach())
        x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
        x = torch.clamp(x, 0, 1)
    
    return x


def in_class(predict, label):

    probs = torch.zeros(10)
    for i in range(10):
        in_class_id = torch.tensor(label == i, dtype= torch.float)
        correct_predict = torch.tensor(predict == label, dtype= torch.float)
        in_class_correct_predict = (correct_predict) * (in_class_id)
        acc = torch.sum(in_class_correct_predict).item() / torch.sum(in_class_id).item()
        probs[i] = acc

    return probs



def match_weight(label, diff0, diff1, diff2):

    weight0 = torch.zeros(label.shape[0], device='cuda')
    weight1 = torch.zeros(label.shape[0], device='cuda')
    weight2 = torch.zeros(label.shape[0], device='cuda')

    for i in range(10):
        weight0 += diff0[i] * torch.tensor(label == i, dtype= torch.float).cuda()
        weight1 += diff1[i] * torch.tensor(label == i, dtype= torch.float).cuda()
        weight2 += diff2[i] * torch.tensor(label == i, dtype= torch.float).cuda()

    return weight0, weight1, weight2



def cost_sensitive(lam0, lam1, lam2):  

    ll0 = torch.clone(lam0)
    ll1 = torch.clone(lam1)

    diff0 = torch.ones(10) * 1 / 10
    for i in range(10):
        for j in range(10):
            if j == i:
                diff0[i] = diff0[i] + 9 / 10 * ll0[i]
            else:
                diff0[i] = diff0[i] - 1 / 10 * ll0[j]

    diff1 = torch.ones(10) * 1/ 10
    for i in range(10):
        for j in range(10):
            if j == i:
                diff1[i] = diff1[i] + 9 / 10 * ll1[i]
            else:
                diff1[i] = diff1[i] - 1 / 10 * ll1[j]

    diff2 = torch.clamp(torch.exp(2 * lam2), min = 0.98, max = 3)

    return diff0, diff1, diff2


def evaluate(model, test_loader, configs, device, class_name, mode = 'Test'):

    print('Doing evaluation mode ' + mode)
    model.eval()

    correct = 0
    correct_adv = 0
    correct_targ_adv = 0
    adv_test_loss = 0 
    total = 0 


    all_label = []
    loss_per_batch =[]
    all_pred = []
    all_pred_adv = []
    all_targ_pred = []
    clean_pred = []
    adv_pred = []
    adv_targ_pred = []
    true_label = []

    for batch_idx, (data, target) in enumerate(test_loader):

        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        all_label.append(target)
        true_label.extend(target.cpu().numpy())
        total += target.size(0)

        ## clean test
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add = pred.eq(target.view_as(pred)).sum().item()
        correct += add
        model.zero_grad()
        all_pred.append(pred)
        print(f'Total clean accuracy {correct/total}')
        clean_pred.extend(pred.cpu().numpy())

        ## adv untargeted test

        x_adv = pgd_linf(model, X = data, y = target, epsilon=configs['epsilon'], alpha=configs['step_size'], num_iter=configs['num_steps'], randomize=False)
        #x_adv = pgd_attack(model, X = data, y = target, epsilon=configs['epsilon'], clip_max=configs['clip_max'],clip_min=configs['clip_min'],num_steps=configs['num_steps'], step_size=configs['step_size'])
        output1 = model(data+x_adv)
        pred1 = output1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add1 = pred1.eq(target.view_as(pred1)).sum().item()
        correct_adv += add1
        all_pred_adv.append(pred1)
        print(f'Total untargeted accuracy {correct_adv/total}')
        adv_pred.extend(pred1.cpu().numpy())

        # Adversarial targeted test
        target_classes = torch.randint(0, len(class_name), target.shape).to(device)  # Example of generating random target classes
        x_adv = pgd_linf_targeted(model, x_natural=data, y_targ=target_classes, epsilon=configs['epsilon'], alpha=configs['step_size'], k=configs['num_steps'], device=device)
        output2 = model(x_adv)
        pred2 = output2.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_targ_adv += pred2.eq(target.view_as(pred2)).sum().item()
        all_targ_pred.append(pred2)
        print(f'Total Targeted accuracy {correct_targ_adv/total}')
        adv_targ_pred.extend(pred2.cpu().numpy())

    all_label = torch.cat(all_label).flatten()
    all_pred = torch.cat(all_pred).flatten()
    all_pred_adv = torch.cat(all_pred_adv).flatten()
    all_targ_pred = torch.cat(all_targ_pred).flatten()

    acc = in_class(all_pred, all_label)
    acc_adv = in_class(all_pred_adv, all_label)
    targ_acc_adv = in_class(all_targ_pred,all_label)

    total_clean_error = 1- correct / len(test_loader.dataset)
    total_bndy_error = correct / len(test_loader.dataset) - correct_adv / len(test_loader.dataset)

    class_clean_error = 1 - acc
    class_bndy_error = acc - acc_adv

    # Clean Confusion Matrix
    clean_conf_matrix = confusion_matrix(clean_pred, true_label)
    clean_df_cm = pd.DataFrame(clean_conf_matrix / np.sum(clean_conf_matrix, axis=1)[:, None], index = [i for i in class_name],
                     columns = [i for i in class_name])
    
    # Untargetd attack Confusion Matrix
    UnTarg_conf_matrix = confusion_matrix(adv_pred, true_label)
    Untarg_df_cm = pd.DataFrame(UnTarg_conf_matrix / np.sum(UnTarg_conf_matrix, axis=1)[:, None], index = [i for i in class_name],
                     columns = [i for i in class_name])
    
    # targetd attack Confusion Matrix
    Targ_conf_matrix = confusion_matrix(adv_targ_pred, true_label)
    targ_df_cm = pd.DataFrame(Targ_conf_matrix / np.sum(Targ_conf_matrix, axis=1)[:, None], index = [i for i in class_name],
                     columns = [i for i in class_name])


    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sn.heatmap(clean_df_cm, annot=True, cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Clean Predictions')

    plt.subplot(1, 2, 2)
    sn.heatmap(Untarg_df_cm, annot=True, cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Untargeted Attacks predictions ')

    plt.tight_layout()

    plt.savefig('confusion_matrices.png')
    plt.show()

    # Plot Targeted Attack Confusion Matrix
    plt.figure(figsize=(9, 7))
    sn.heatmap(targ_df_cm, annot=True, cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Targeted Attacks Predictions')
    plt.savefig('targeted_confusion_matrix.png')
    plt.show()



    return class_clean_error, class_bndy_error, total_clean_error, total_bndy_error





def trades_adv(model,
               x_natural,
               weight,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size):

    # define KL-loss
    new_eps = (epsilon * weight).view(weight.shape[0], 1, 1, 1)
    step_size = torch.max(new_eps) / 4

    criterion_kl = nn.KLDivLoss(size_average = False)
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - new_eps), x_natural + new_eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv




def train_model(model, train_loader, optimizer, diff0, diff1, diff2, epoch, beta, configs, device):

    criterion_kl = nn.KLDivLoss(reduction='none')
    criterion_nat = nn.CrossEntropyLoss(reduction='none')

    print('Doing Training on epoch:  ' + str(epoch))

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)

        weight0, weight1, weight2 = match_weight(target, diff0, diff1, diff2)
        ## generate adv examples
        x_adv = trades_adv(model, x_natural = data, weight = weight2, **configs)

        model.train()
        ## clear grads
        optimizer.zero_grad()

        ## get loss
        loss_natural = criterion_nat(model(data), target)
        loss_bndy_vec = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(data), dim=1))
        loss_bndy = torch.sum(loss_bndy_vec, 1)

        ## merge loss
        loss = torch.sum(loss_natural * weight0)/ torch.sum(weight0)  + beta * torch.sum(loss_bndy * weight1) / torch.sum(weight1)        ## back propagates
        loss.backward()
        optimizer.step()

        ## clear grads
        optimizer.zero_grad()



def frl_train(h_net, ds_train, ds_valid, optimizer, class_name, now_epoch, configs, configs1, device, delta0, delta1, rate1, rate2, lmbda, beta, lim):
    print('train epoch ' + str(now_epoch), flush=True)

    ## given model, get the validation performance and gamma
    class_clean_error, class_bndy_error, total_clean_error, total_bndy_error = \
        evaluate(h_net, ds_valid, configs1, device,class_name, mode='Validation')

    ## get gamma on validation set
    gamma0 = class_clean_error - total_clean_error - delta0
    gamma1 = class_bndy_error - total_bndy_error - delta1

    ## print inequality results
    print('total clean error ' + str(total_clean_error))
    print('total boundary error ' + str(total_bndy_error))

    print('.............')
    print('each class inequality constraints')
    print(gamma0)
    print(gamma1)

    #################################################### do training on now epoch
    ## constraints coefficients
    lmbda0 = lmbda[0:10] + rate1 * torch.clamp(gamma0, min = -1000)      ## update langeragian multiplier
    lmbda1 = lmbda[10:20] + rate1 * 2 * torch.clamp(gamma1, min = -1000)      ## update langeragian multiplier
    lmbda2 = lmbda[20:30] + rate2 * gamma1

    lmbda0 = normalize_lambda(lmbda0, lim)
    lmbda1 = normalize_lambda(lmbda1, lim)   ## normalize back to the simplex

    ## given langerangian multipliers, get each class's weight
    lmbda = torch.cat([lmbda0, lmbda1, lmbda2])
    diff0, diff1, diff2 = cost_sensitive(lmbda0, lmbda1, lmbda2)

    print('..............................')
    print('current weight')
    print(diff0)
    print(diff1)
    print(diff2)
    print('..............................')
    ## do the model parameter update based on gamma
    _ = train_model(h_net, ds_train, optimizer, diff0, diff1, diff2, now_epoch,
                    beta, configs, device)

    return lmbda


def normalize_lambda(lmb, lim = 0.8):

    lmb = torch.clamp(lmb, min=0)
    if torch.sum(lmb) > lim:
        lmb = lim * lmb / torch.sum(lmb)
    else:
        lmb = lmb
    return lmb

