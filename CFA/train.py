import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from attack import pgd_loss, cw_pgd_loss, trades_loss, cw_trades_loss, fat_loss, cw_fat_loss, pgd_linf_targeted
from utils import dev, normalize_cifar, load_valid_dataset, weight_average
from model import PreActResNet18
from resnet import ResNet18
from wide_resnet import WideResNet
from sklearn.metrics import confusion_matrix
import seaborn as sn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--model', default='RES18', type=str, choices=['PRN', 'WRN', 'RES18']) 
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--mode', default='TRADES', type=str, choices=['AT', 'TRADES', 'FAT'])

    parser.add_argument('--epsilon', default=8, type=float)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=int)
    parser.add_argument('--norm', default='Linf', type=str)

    parser.add_argument('--beta', default=6, type=int)  # beta for TRADES
    parser.add_argument('--tau', default=3, type=int)   #  tau for FAT -> It automatically adjusts attatck strength on each instance  

    parser.add_argument('--fname', type=str, default='auto') #TODO
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--ccm', action='store_true') # CCM
    parser.add_argument('--ccr', action='store_true') # CCR
    parser.add_argument('--lambda-1', default=0.5, type=float)
    parser.add_argument('--lambda-2', default=0.5, type=float)

    parser.add_argument('--begin', default=10, type=int)

    parser.add_argument('--decay-rate', default=0.88 ,type=float)
    parser.add_argument('--threshold', default=0.24, type=float)

    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

class CW_log():
    def __init__(self, class_num = 10) -> None:
        self.N = 0
        self.robust_acc = 0
        self.clean_acc = 0
        self.cw_robust = torch.zeros(class_num).to(device)
        self.cw_clean = torch.zeros(class_num).to(device)
        self.class_num = class_num
        self.clean_correct = []  # List to store correct predictions for clean data
        self.robust_correct = []  # List to store correct predictions for robust data
    
    def update_clean(self, output, y):
        self.N += len(output)
        clean_pred = output.max(1)[1]
        clean_correct = clean_pred == y
        self.clean_acc += clean_correct.sum()
        
        # Store correct predictions
        self.clean_correct.extend(clean_correct)
        
        for i, c in enumerate(y):
            if clean_correct[i]:
                self.cw_clean[c] += 1
    
    def update_robust(self, output, y):
        robust_pred = output.max(1)[1]
        robust_correct = robust_pred == y
        self.robust_acc += robust_correct.sum()
        
        # Store correct predictions
        self.robust_correct.extend(robust_correct)
        
        for i, c in enumerate(y):
            if robust_correct[i]:
                self.cw_robust[c] += 1
    
    def result(self):
        N = self.N
        m = self.class_num
        clean_correct_array = torch.tensor(self.clean_correct).float().cpu().numpy()
        robust_correct_array = torch.tensor(self.robust_correct).float().cpu().numpy()
        return (self.clean_acc/N, 
                self.robust_acc/N, 
                m*self.cw_clean/N, 
                m*self.cw_robust/N,
                clean_correct_array,  # Return clean_correct values
                robust_correct_array)  # Return robust_correct values



def train_epoch(model, loader, opt, device, attack, eps, beta, alpha, n_iters):
    model.train()
    print('started model training')
    logger = CW_log()
   
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        loss, output = attack(model,x,y,eps,beta,alpha,n_iters)
        opt.zero_grad()
        loss.backward()
        opt.step()

        logger.update_robust(output, y)

        clean_output = model(normalize_cifar(x)).detach() # removed normalized input
        logger.update_clean(clean_output, y)
        if args.debug:
            break
    return logger.result()

def eval_epoch(model, loader, device, class_name, attack, eps, beta, alpha, n_iters, mode='Test'):
    model.eval()
    print('Doing Untargeted evaluation mode ' + mode)
    correct = 0
    True_label = []
    Predicted = []
    logger = CW_log()
    
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        _, output = attack(model,x,y,eps,beta,alpha,n_iters)
        _, predicted = output.max(1)
        correct += predicted.eq(y).sum().item()
        True_label.extend(y.cpu().numpy())
        Predicted.extend(predicted.cpu().numpy())
        logger.update_robust(output, y)

        clean_output = model(normalize_cifar(x)).detach()   # removed normalized input
        logger.update_clean(clean_output, y)
        if args.debug:
            break

    return logger.result(), True_label, Predicted

def clean_evaluate(model, loader, device, class_names, mode='Test'):
    print('Doing evaluation mode ' + mode)
    model.eval()

    correct = 0
    total = 0
    clean_pred = []
    true_label = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Evaluating")):
            data, target = data.to(device), target.to(device)
            total += target.size(0)

            true_label.extend(target.cpu().numpy())

            # Clean test
            outputs = model(normalize_cifar(data)).detach()     # removed normalized input

            _, clean_predicted = outputs.max(1)  # Get the index of the max log-probability
            correct += clean_predicted.eq(target).sum().item()
            clean_pred.extend(clean_predicted.cpu().numpy())

    accuracy = 100. * correct / total
    print(f'{mode} Accuracy: {accuracy:.2f}%')

    return clean_pred, true_label

def targeted_evaluate(model, loader, device, class_names, attack, eps, alpha, n_iters, mode='Targeted Test'):
    model.eval()
    print('Doing Targeted evaluation mode ' + mode)
    true_labels = []
    targeted_labels = []
    targeted_predictions = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Evaluating Targeted Attack")):
            data, target = data.to(device), target.to(device)

        # Generate random target classes for targeted attack
            y_targ = torch.randint(0, len(class_names), target.shape).to(device)


        # Generate adversarial examples
            x_adv_targ = pgd_linf_targeted(model, data, y_targ, eps, alpha, n_iters, device)
        
        # Model inference
    
            targ_outputs = model(normalize_cifar(x_adv_targ)).detach()
            _, targ_predicted = targ_outputs.max(1)

            true_labels.extend(target.cpu().numpy())
            targeted_labels.extend(y_targ.cpu().numpy())
            targeted_predictions.extend(targ_predicted.cpu().numpy())

    return true_labels, targeted_labels, targeted_predictions

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

def lr_schedule(t):
    if t / args.epochs < 0.5:
        return args.lr_max
    elif t / args.epochs < 0.75:
        return args.lr_max / 10.
    else:
        return args.lr_max / 100.

def lr_schedule_wrn(t):
    if t < 75:
        return args.lr_max
    elif t < 90:
        return args.lr_max / 10.
    else:
        return args.lr_max / 100.
    
if __name__ == '__main__':
    args = get_args()
    if args.fname == 'auto':
        args.fname = f'cifar10_{args.model}_{args.mode}{"_ccm" if args.ccm else ""}{"_ccr" if args.ccr else ""}'
    fname = args.fname
    device = dev(args.device)
    eps = args.epsilon / 255.       # 8/255
    alpha = args.pgd_alpha / 255.   # 2/255
    beta = args.beta / 1.           # 6
    class_eps = torch.ones(10).to(device) * eps
    class_beta = torch.ones(10).to(device) * (beta/(1+beta))
    iteration = args.attack_iters
    epochs = args.epochs if args.model == 'PRN' else 100

    train_loader, valid_loader, test_loader , class_names = load_valid_dataset('cifar10')
    
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('logs'):
        os.mkdir('logs')
        
    if not os.path.exists('models/'+args.fname):
        os.mkdir('models/'+args.fname)
    if not os.path.exists('logs/'+args.fname):
        os.mkdir('logs/'+args.fname)
    with open(f'logs/{fname}/config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    if args.model == 'PRN':
        model = PreActResNet18().to(device)
    elif args.model == 'WRN':
        model = WideResNet().to(device)
    elif args.model == 'RES18':
        model = ResNet18().to(device)
    else:
        raise ValueError
    
    # init weight averaged model
    EMA_model = PreActResNet18().to(device) if args.model == 'PRN' else ResNet18().to(device) # Exponential Moving Average
    FAWA_model = PreActResNet18().to(device) if args.model == 'PRN' else ResNet18().to(device) # Fairness average weight averaging
    EMA_model.eval()
    FAWA_model.eval()  # FAWA method is a variant of weight average method with Exponential Moving Average

    SEAT_init = False
    
    params = model.parameters()
    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    log_data = [] # Epochs * 7:    Epoch, train_clean, train, valid_clean, valid, test_clean, test
    cw_data = []  # Epochs * 6 * 10:    Epoch, min-{train_clean, train, valid_clean, valid, test_clean, test}
    EMA_log, FAWA_log = [], []
    save_threshold = [0, 0, 0] # robust+min_robust, for main, EMA, FAWA
    for epoch in range(epochs):
       
    # update learning rate
        if args.model == 'WRN':
            lr = lr_schedule_wrn(epoch)
        else:
            lr = lr_schedule(epoch)
        opt.param_groups[0].update(lr=lr)
    
    # train
        model.train()
        # ccm
        if epoch >= args.begin:
            cw_tensor = cw_tensor.to(device)
            train_robust = cw_tensor[-1, 1, :]
            class_eps = (torch.ones(10).to(device) * args.lambda_1 + train_robust) * eps
        else:
            class_eps = torch.ones(10).to(device) * eps
    
        # ccr
        if args.ccr and epoch >= args.begin:
            for i in range(10):
                class_beta[i] = (args.lambda_2+train_robust[i]) * beta / (1 + (args.lambda_2+train_robust[i])*beta)
        else:
            class_beta = torch.ones(10).to(device) * (beta/(1+beta))

        # set tau for FAT
        if args.mode == 'FAT':
            class_beta = args.tau
        
        if args.mode == 'AT':
            if args.ccm:
                attack = cw_pgd_loss
            else:
                attack = pgd_loss
        elif args.mode == 'TRADES':
            if args.ccm:
                attack = cw_trades_loss
            else:
                attack = trades_loss
        elif args.mode == 'FAT':
            if args.ccm:
                attack = cw_fat_loss
            else:
                attack = fat_loss
        


        if args.ccm:
            train_result = train_epoch(model, train_loader, opt, device, attack, class_eps, class_beta, alpha, iteration)
        else:
            train_result = train_epoch(model, train_loader, opt, device, attack, eps, class_beta, alpha, iteration)
        
        model.eval()
        test_result, True_label, Predicted = eval_epoch(model, test_loader, device, class_names, pgd_loss, 8./255., beta, 2./255., 10)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            test_result, True_label, Predicted = eval_epoch(model, test_loader, device, class_names, pgd_loss, 8./255., beta, 2./255., 10)
            # Extract results
            clean_accuracy, robust_accuracy, cw_clean, cw_robust, clean_correct, robust_correct = test_result
            # clean accuracy

            #clean_pred, True_label = clean_evaluate(model, test_loader, device, class_names, mode = 'Test')
            #clean_conf_matrix = confusion_matrix(True_label, clean_pred)
            #robust_conf_matrix = confusion_matrix(True_label, Predicted)
            true_labels_adv, targeted_labels_adv, adv_predictions = targeted_evaluate(model, test_loader, device, class_names, pgd_loss, 8./255., alpha, args.attack_iters, mode='Targeted Test')
            conf_matrix_adv = confusion_matrix(true_labels_adv, adv_predictions)

            #plot_confusion_matrix(clean_conf_matrix, class_names, title='Confusion Matrix for Untargeted Training using CFA and Clean Predictions')
            #plot_confusion_matrix(robust_conf_matrix, class_names, title='Confusion Matrix for Untargeted Training using CFA and Untargeted Predictions')
            plot_confusion_matrix(conf_matrix_adv, class_names, 'Confusion Matrix for Untargeted Training using CFA and targeted Predictions')



    
        # valid
        valid_result, _ , _ = eval_epoch(model, valid_loader, device,class_names, pgd_loss, 8./255., beta, 2./255., 10)
        
        # weight average
        # EMA (Exponential Moving Average) -> just a weight averaging method
        weight_average(EMA_model, model, args.decay_rate, epoch==0)
        EMA_result, _ , _ = eval_epoch(EMA_model, test_loader, device, class_names, pgd_loss, 8./255., beta, 2./255., 10)

        # FAWA
        R_min = valid_result[3].min()
        if R_min >= args.threshold:
            if not SEAT_init:
                SEAT_init = True
                weight_average(FAWA_model, model, args.decay_rate, True)
            else:
                weight_average(FAWA_model, model, args.decay_rate, False)
        else:
            weight_average(FAWA_model, model, 1., False)
        FAWA_result, _ , _ = eval_epoch(FAWA_model, test_loader, device,class_names, pgd_loss, 8./255., beta, 2./255., 10)

        # log result
        log_data.append(torch.tensor([epoch, train_result[0], train_result[1], 
        valid_result[0], valid_result[1], test_result[0], test_result[1]]))

        cw_data.append(torch.stack([train_result[2], train_result[3], 
        valid_result[2], valid_result[3], test_result[2], test_result[3]], dim=0))

        log_tensor = torch.stack(log_data, dim=0).cpu() # Epochs * 7
        cw_tensor = torch.stack(cw_data, dim=0).cpu()   # Epochs * 6 * 10

        torch.save(log_tensor, f'models/{args.fname}/log.pth')
        torch.save(cw_tensor, f'models/{args.fname}/cw_log.pth')

        # plot
        log_arr = log_tensor.numpy()
        cw_arr = cw_tensor.min(2)[0].numpy()
        log_arr = np.concatenate([log_arr, cw_arr], axis=1)
        report_arr = np.concatenate([log_arr[:, 5:7], cw_arr[:, 4:]], axis=1) # clean, robust, min-clean, min-robust
        df = pd.DataFrame(log_arr)
        df.to_csv(f'logs/{args.fname}/log.csv')
        df = pd.DataFrame(report_arr)
        df.to_csv(f'logs/{args.fname}/report_log.csv')

        EMA_log.append([
            torch.tensor([EMA_result[0], EMA_result[1], EMA_result[2].min(), EMA_result[3].min()]).cpu().numpy()
        ])
        FAWA_log.append([
            torch.tensor([FAWA_result[0], FAWA_result[1], FAWA_result[2].min(), FAWA_result[3].min()]).cpu().numpy()
        ])

        EMA_data = np.concatenate(EMA_log, axis=0)
        FAWA_data = np.concatenate(FAWA_log, axis=0)

        df = pd.DataFrame(EMA_data)
        df.to_csv(f'logs/{args.fname}/EMA_log.csv')

        df = pd.DataFrame(FAWA_data)
        df.to_csv(f'logs/{args.fname}/FAWA_log.csv')

        state = {
        'net': model.state_dict()
        }

        file_path = './save_model/'
        if not os.path.isdir(file_path):
            os.mkdir(file_path)

        # save models
        if epoch >= 0.5 * args.epochs:
            # Main
            index = log_tensor[-1,-1] + cw_tensor[-1, -1].min()
            if index >= save_threshold[0] - 0.02 or epoch >= args.epochs-5:
                torch.save(model.state_dict(), f'models/{args.fname}/{epoch}.pth')
                torch.save(state, file_path + args.fname)
                print('Model Saved!')
                save_threshold[0] = max(save_threshold[0], index.item())
            
            # EMA
            index = EMA_data[-1, 1] + EMA_data[-1, 3]
            if index >= save_threshold[1] - 0.02 or epoch >= args.epochs-5:
                torch.save(EMA_model.state_dict(), f'models/{args.fname}/EMA_{epoch}.pth')
                save_threshold[1] = max(save_threshold[1], index.item())

            # FAMA
            index = FAWA_data[-1, 1] + FAWA_data[-1, 3]
            if index >= save_threshold[2] - 0.02 or epoch >= args.epochs-5:
                torch.save(FAWA_model.state_dict(), f'models/{args.fname}/FAWA_{epoch}.pth')
                save_threshold[2] = max(save_threshold[2], index.item())