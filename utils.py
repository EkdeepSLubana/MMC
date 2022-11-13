import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import numpy as np
cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


### LR Scheduler
class LR_Scheduler(object):
    def __init__(self, optimizer, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        decay_iter = iter_per_epoch * num_epochs
        self.lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))        
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr

    def get_lr(self):
        return self.current_lr


### Linear eval
def linear_eval(net, dataloader, suffix="nocue", return_loss=False, **kwargs):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        with tqdm(dataloader, unit=" batch") as tepoch:
            batch_idx = 0
            for inputs, targets in tepoch:
                tepoch.set_description(f"Test {suffix}")
                batch_idx += 1

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tepoch.set_postfix(loss=test_loss/batch_idx, accuracy=100. * correct/total) 

    if return_loss:
        return 100. * correct/total, test_loss/batch_idx
    else:
        return [100. * correct/total]


### Change setup
def change_setup(base_setup, attr_change=[], attr_val=[]):
    setup = base_setup.copy()
    for attr, val in zip(attr_change, attr_val):
        setup[attr] = val
    return setup