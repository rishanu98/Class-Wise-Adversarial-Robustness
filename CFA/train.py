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
from attack import pgd_loss, cw_pgd_loss, trades_loss, cw_trades_loss, fat_loss, cw_fat_loss
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
    parser.add_argument('--model', default='RES18', type=str, choices=['PRN', 'WRN', 'RES18']) #
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--mode', default='TRADES', type=str, choices=['AT', 'TRADES', 'FAT'])

    parser.add_argument('--epsilon', default=8, type=int)
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
        self.cw_robust = torch.zeros(10).to(device)
        self.cw_clean = torch.zeros(10).to(device)
        self.class_num = class_num
    
    def update_clean(self, output, y):
        self.N += len(output)
        pred = output.max(1)[1]
        correct = pred == y
        self.clean_acc += correct.sum()

        for i, c in enumerate(y):
            if correct[i]:
                self.cw_clean[c] += 1
    
    def update_robust(self, output, y):
        pred = output.max(1)[1]
        correct = pred == y
        self.robust_acc += correct.sum()

        for i, c in enumerate(y):
            if correct[i]:
                self.cw_robust[c] += 1
    
    def result(self):
        N = self.N
        m = self.class_num
        return self.clean_acc/N, self.robust_acc/N, m*self.cw_clean/N, m*self.cw_robust/N



def train_epoch(model, loader, opt, device, attack, eps, beta, alpha, n_iters):
    model.train()
    logger = CW_log()
    loader = tqdm(loader)
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        loss, output = attack(model,x,y,eps,beta,alpha,n_iters)
        opt.zero_grad()
        loss.backward()
        opt.step()

        logger.update_robust(output, y)

        clean_output = model(normalize_cifar(x)).detach()
        logger.update_clean(clean_output, y)
        if args.debug:
            break
    return logger.result()

def eval_epoch(model, loader, device, class_name, attack, eps, beta, alpha, n_iters):
    model.eval()
    correct = 0
    True_label = []
    Predicted = []
    logger = CW_log()
    loader = tqdm(loader)
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        _, output = attack(model,x,y,eps,beta,alpha,n_iters)
        _, predicted = output.max(1)
        correct += predicted.eq(y).sum().item()
        True_label.extend(y.cpu().numpy())
        Predicted.extend(predicted.cpu().numpy())
        logger.update_robust(output, y)

        clean_output = model(normalize_cifar(x)).detach()
        logger.update_clean(clean_output, y)
        if args.debug:
            break

    return logger.result(), True_label, Predicted

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
    EMA_model = PreActResNet18().to(device) if args.model == 'PRN' else WideResNet().to(device) # Exponential Moving Average
    FAWA_model = PreActResNet18().to(device) if args.model == 'PRN' else WideResNet().to(device) # Fairness average weight averaging
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
        # test
        test_result, True_label, Predicted = eval_epoch(model, test_loader, device, class_names, pgd_loss, 8./255., beta, 2./255., 10)
        conf_mat = confusion_matrix(Predicted,True_label)
        row_sums = np.sum(conf_mat, axis=1)
        row_sums[row_sums == 0] = 1e-10

        targ_df_cm = pd.DataFrame(conf_mat / row_sums[:,None], index = [i for i in class_names],
                     columns = [i for i in class_names])
        plt.figure(figsize=(8, 6))
        sn.heatmap(targ_df_cm, annot=True, cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix for TRADES_PRN_CCR model')


        plt.tight_layout()

        plt.savefig('TRADES_PRN_CCR.png')

        plt.show()
    
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

        # save models
        if epoch >= 0.5 * args.epochs:
            # Main
            index = log_tensor[-1,-1] + cw_tensor[-1, -1].min()
            if index >= save_threshold[0] - 0.02 or epoch >= args.epochs-5:
                torch.save(model.state_dict(), f'models/{args.fname}/{epoch}.pth')
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