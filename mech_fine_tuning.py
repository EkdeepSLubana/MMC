import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pickle as pkl
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import copy

import syndata
from models import create_model
from utils import linear_eval, LR_Scheduler

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


### LLRT
def LLRT(dataloader, n_epochs=25, base_lr=1):
    criterion = nn.CrossEntropyLoss()

    net_nc.eval()
    c_shape = net_nc.linear.weight.shape
    classifier = nn.Linear(in_features=c_shape[1], out_features=c_shape[0], bias=True).to(device)
    w_start = classifier.weight.data.clone().detach()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=base_lr, momentum=0.9, weight_decay=0)
    lr_scheduler = utils.LR_Scheduler(optimizer, num_epochs=n_epochs, base_lr=base_lr, final_lr=0, iter_per_epoch=len(dataloader))

    for epoch in range(n_epochs):
        train_loss, correct, total = 0, 0, 0
        with tqdm(dataloader, unit=" batch") as tepoch:
            batch_idx = 0
            for inputs, targets in tepoch:
                tepoch.set_description(f"Train Epoch {epoch}")
                batch_idx += 1
                
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = classifier(net_nc(inputs, use_linear=False))
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

                tepoch.set_postfix(loss=train_loss/batch_idx, accuracy=100. * correct/total, lr=optimizer.param_groups[0]['lr'], 
                                    Delta=(classifier.weight.data.detach() - w_start).norm().item())

    net_nc.linear.weight.data = classifier.weight.data.clone()
    net_nc.linear.bias.data = classifier.bias.data.clone()


### Naive Fine-Tuning
def fine_tuning(dataloader, epoch):
    criterion = nn.CrossEntropyLoss()
    net_nc.train()
    train_loss, correct, total = 0, 0, 0
    
    with tqdm(dataloader, unit=" batch") as tepoch:
        batch_idx = 0
        for inputs, targets in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}")
            batch_idx += 1
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_nc.zero_grad()
            outputs = net_nc(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_nc.step()
            lr_scheduler.step() 

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            tepoch.set_postfix(loss=train_loss/batch_idx, accuracy=100. * correct/total, lr=optimizer_nc.param_groups[0]['lr'])

    return 100. * correct/total, 0


### CBFT
def CBFT(dataloader_cue, dataloader_nocue, epoch=0, lambd=1, warmup_epochs=1, loss_margin=1.0):

    net_nc.train()
    criterion = nn.CrossEntropyLoss()
    train_loss_c, train_loss_nc, correct_c, correct_nc, total_c, total_nc = 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8
    n_classes = net_nc.linear.weight.shape[0]

    with tqdm(zip(dataloader_nocue, dataloader_cue), unit=" batch") as tepoch:
        batch_idx = 0
        for (inputs_nc, targets_nc), (inputs_c, targets_c) in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}")
            batch_idx += 1
            
            # data
            inputs_c, targets_c = inputs_c.to(device), targets_c.to(device)
            inputs_nc, targets_nc = inputs_nc.to(device), targets_nc.to(device)

            # create a class-data dict for invariance loss
            class_ids_nc = {i: (targets_nc == i).nonzero().view(-1,1).type(torch.long) for i in range(n_classes)} 
            class_ids_c = {i: (targets_c == i).nonzero().view(-1,1).type(torch.long) for i in range(n_classes)}

            ##### Step 1 execute: Barrier loss!
            # get linear interpolated model
            with torch.no_grad():
                # pick random point on path
                t = 0.5 + 0.5 * (torch.randn(1)).clip(-0.5, 0.5)[0] 

                for module_c, module_nc, module_interpolated in zip(net_c.modules(), net_nc.modules(), net_interpolated.modules()):
                    if(isinstance(module_c, nn.Conv2d)):
                        module_interpolated.weight.data = t * module_c.weight.data + (1-t) * module_nc.weight.data
                    elif(isinstance(module_c, nn.BatchNorm2d)):
                        module_interpolated.weight.data = t * module_c.weight.data + (1-t) * module_nc.weight.data
                        module_interpolated.bias.data = t * module_c.bias.data + (1-t) * module_nc.bias.data
                    elif(isinstance(module_c, nn.Linear)):
                        module_interpolated.weight.data = t * module_c.weight.data + (1-t) * module_nc.weight.data
                        module_interpolated.bias.data = t * module_c.bias.data + (1-t) * module_nc.bias.data
                    else:
                        pass

            # outputs from the interpolated model
            net_interpolated.zero_grad()
            outputs_interpolated = net_interpolated(inputs_c)

            # compute loss
            loss = (loss_margin - criterion(outputs_interpolated, targets_c)).abs()
            loss.backward()

            # associate grads from interpolated model to theta_nc
            with torch.no_grad():
                for module_nc, module_interpolated in zip(net_nc.modules(), net_interpolated.modules()):
                    if(isinstance(module_nc, nn.Conv2d)):
                        module_nc.weight.grad = (1-t) * module_interpolated.weight.grad.data
                    elif(isinstance(module_nc, nn.BatchNorm2d)):
                        module_nc.weight.grad = (1-t) * module_interpolated.weight.grad.data
                        module_nc.bias.grad = (1-t) * module_interpolated.bias.grad.data
                    elif(isinstance(module_nc, nn.Linear)):
                        module_nc.weight.grad = (1-t) * module_interpolated.weight.grad.data
                        module_nc.bias.grad = (1-t) * module_interpolated.bias.grad.data
                    else:
                        pass

            # Step 1 update
            optimizer_nc.step()

            train_loss_c += loss.item()
            _, predicted_c = outputs_interpolated.max(1)
            correct_c += predicted_c.eq(targets_c).sum().item()
            total_c += targets_c.size(0)


            ##### Step 2: No-cue + Invariance loss
            # compute loss
            net_nc.zero_grad()
            if(epoch < warmup_epochs):
                outputs_ns = net_ns(inputs_ns)
                loss = criterion(outputs_ns, targets_ns)
            else:
                z_nc, z_c = net_nc(inputs_nc, use_linear=False), net_nc(inputs_c, use_linear=False)
                inv_loss = 0
                for i in range(n_classes):
                    if (class_ids_nc[i].shape[0] == 0 or class_ids_c[i].shape[0] == 0):
                        continue
                    # MSE
                    inv_loss += (F.normalize(z_nc[class_ids_nc[i][:,0]].mean(dim=0, keepdim=True), dim=1) - F.normalize(z_c[class_ids_c[i][:,0]].mean(dim=0, keepdim=True), dim=1)).norm().pow(2)

                outputs_nc = net_nc.linear(z_nc)
                loss = criterion(outputs_nc, targets_nc) + lambd * inv_loss / n_classes

            # Step 2 update
            loss.backward()
            optimizer_nc.step()
            if(epoch > warmup_epochs):
                lr_scheduler.step()

            train_loss_nc += loss.item()
            _, predicted_nc = outputs_nc.max(1)
            correct_nc += predicted_nc.eq(targets_nc).sum().item()
            total_nc += targets_nc.size(0)

            torch.cuda.empty_cache()

            tepoch.set_postfix(loss_c=train_loss_c/batch_idx, accuracy_c=100. * correct_c/total_c, loss_nc=train_loss_nc/batch_idx, 
                                    accuracy_nc=100. * correct_nc/total_nc, lr=optimizer_nc.param_groups[0]['lr'])

    return 100. * correct_c/total_c, 100. * correct_nc/total_nc

    
    
### Main
if __name__ == "__main__":

    # setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cue_model_path", type=str, default="not provided")
    parser.add_argument("--n_clean", type=int, default=2500)
    parser.add_argument("--n_cue", type=int, default=47500)
    parser.add_argument("--ft_method", type=str, choices=["CBFT", "naive_ft", "LLRT", "LPFT"])
    parser.add_argument("--model", default='VGG-13', choices=['VGG-13', 'res18'])
    parser.add_argument("--base_dataset", default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'Dominoes'])
    parser.add_argument("--cue_type", choices=['box', 'dominoes'])
    parser.add_argument("--cue_proportion", type=float, default=1.)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_lr", type=float, default=0.01)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--save_proportion", type=float, default=0.6)
    args = parser.parse_args()

    if(args.cue_model_path == 'not provided'):
        raise Exception("Need a pretrained model trained on data with cues")

    torch.manual_seed(args.seed)
    print("-- Model:", args.model)
    print("-- Dataset:", args.base_dataset)
    print("-- Using pretrained model from path", args.cue_model_path)
    if (args.cue_type != 'nocue'):
        print("-- Cue type:", args.cue_type)
        print("-- Cue proportion:", args.cue_proportion)
    else:
        print("-- Training without cues")
    print("-- Training epoch:", args.n_epochs)
    print("-- Model saved after training:", args.save_model)
    print("\n")



    ##### dataloaders
    # no cue dataloader (we're kinda hardcoding here that the dataset size is 50000--will fix later)
    subset_ids = np.random.randint(0, 50000, args.n_clean) if args.n_clean < 50000 else None
    dataloader_nocue = syndata.get_dataloader(load_type='train', base_dataset=args.base_dataset, cue_type='nocue', 
                                                batch_size=args.batch_size, subset_ids=subset_ids)
    if(args.ft_method == 'CBFT'):
        subset_ids = np.random.randint(0, 50000, args.n_cue) if args.n_cue < 50000 else None
        dataloader_cue = syndata.get_dataloader(load_type='train', base_dataset=args.base_dataset, cue_type=args.cue_type, 
                                        cue_proportion=args.cue_proportion, batch_size=args.batch_size, subset_ids=subset_ids)

    nocue_testloader = syndata.get_dataloader(load_type='test', base_dataset=args.base_dataset, cue_type='nocue', batch_size=args.batch_size)

    # cue dataloader
    cue_testloader = syndata.get_dataloader(load_type='test', base_dataset=args.base_dataset, cue_type=args.cue_type, cue_proportion=args.cue_proportion, 
                                        batch_size=args.batch_size)

    # rand. cue dataloader
    rand_cueloader = syndata.get_dataloader(load_type='test', base_dataset=args.base_dataset, cue_type=args.cue_type, cue_proportion=args.cue_proportion, 
                                    batch_size=args.batch_size, randomize_cue=True)

    # rand. image dataloader
    rand_imgloader = syndata.get_dataloader(load_type='test', base_dataset=args.base_dataset, cue_type=args.cue_type, cue_proportion=args.cue_proportion, 
                                        batch_size=args.batch_size, randomize_img=True)


    ##### models
    # cue model
    net_c = create_model(args.model, num_classes=len(nocue_testloader.dataset.classes)).to(device)
    net_c.load_state_dict(torch.load(args.cue_model_path))

    # no cue model
    net_nc = create_model(args.model, num_classes=len(nocue_testloader.dataset.classes)).to(device)
    if args.ft_method != 'train_scratch':
        net_nc = copy.deepcopy(net_c)

    # interpolated model used for CBFT
    if(args.ft_method == 'CBFT'):
        net_interpolated = create_model(args.model, num_classes=len(nocue_testloader.dataset.classes)).to(device)
        net_interpolated.train()


    ##### stats for later analysis
    accs_dict = {'train_cue': [], 'train_nocue': [], 'test_cue': [], 'test_nocue': [], 'test_randcue': [], 'test_randimg': []}


    ##### LLRT / LPFT: do LLRT separately for ease of implementation
    if(args.ft_method == 'LLRT' or args.ft_method == 'LPFT'):
        LLRT(dataloader=dataloader_nocue, n_epochs=200, base_lr=30)
        saved_results = {
                            'nocue': linear_eval(net_nc.eval(), dataloader=nocue_testloader, suffix='nocue'), 
                            'cue': linear_eval(net_nc.eval(), dataloader=cue_testloader, suffix=args.cue_type), 
                            'randcue': linear_eval(net_nc.eval(), dataloader=rand_cueloader, suffix='randcue'), 
                            'randimg': linear_eval(net_nc.eval(), dataloader=rand_imgloader, suffix='randimg')
                        }
        with open("./tab_results/{}_LLRT_{}_proportion_{}_seed_{}.pkl".format(args.model, args.base_dataset, args.save_proportion, args.seed), 'wb') as f:
            pkl.dump(saved_results, f)



    ##### optimizer
    optimizer_nc = optim.SGD(net_nc.parameters(), lr=args.start_lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = utils.LR_Scheduler(optimizer_nc, num_epochs=args.n_epochs, base_lr=args.start_lr, final_lr=0, iter_per_epoch=len(dataloader_nocue))



    ##### train for different methods
    for epoch_num in range(args.n_epochs):

        ### LLRT is done above, so break
        if(args.ft_method == 'LLRT'):
            break

        ### Naive fine-tuning and LPFT: LPFT is just LLRT + naive FT
        elif(args.ft_method == 'naive_ft' or args.ft_method == 'LPFT'):
            acc_c, acc_nc = fine_tuning(dataloader=dataloader_nocue, epoch=epoch_num)

        ### CBFT
        elif(args.ft_method == 'CBFT'):
            acc_c, acc_nc = CBFT(dataloader_cue=dataloader_cue, dataloader_nocue=dataloader_nocue, epoch=epoch_num)

        # update stats
        accs_dict['train_cue'] += [acc_c]
        accs_dict['train_nocue'] += [acc_nc]
        if (epoch_num == args.n_epochs - 1): 
            accs_dict['test_nocue'] += linear_eval(net_nc.eval(), dataloader=nocue_testloader, suffix='nocue')
            accs_dict['test_cue'] += linear_eval(net_nc.eval(), dataloader=cue_testloader, suffix=args.cue_type)
            accs_dict['test_randcue'] += linear_eval(net_nc.eval(), dataloader=rand_cueloader, suffix='randcue')
            accs_dict['test_randimg'] += linear_eval(net_nc.eval(), dataloader=rand_imgloader, suffix='randimg')
        torch.cuda.empty_cache()


    # save stats
    saved_results = {
                        'nocue': accs_dict['test_nocue'][-1], 
                        'cue': accs_dict['test_cue'][-1], 
                        'randcue': accs_dict['test_randcue'][-1], 
                        'randimg': accs_dict['test_randimg'][-1]
                    }
    with open("./tab_results/{}_{}_{}_{}_proportion_{}_seed_{}.pkl".format(args.start_lr, args.model, args.ft_method, args.base_dataset, args.save_proportion, args.seed), 'wb') as f:
        pkl.dump(saved_results, f)
        
