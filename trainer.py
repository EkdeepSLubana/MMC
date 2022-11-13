import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import pickle as pkl
from tqdm import tqdm
import torch.backends.cudnn as cudnn

import syndata
from models import create_model
from utils import linear_eval

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dir for saving models
os.makedirs('./saved_models', exist_ok=True)
os.makedirs('./saved_models/nocue', exist_ok=True)
os.makedirs('./saved_models/cue', exist_ok=True)

# Dir for saving training stats
os.makedirs('./training_stats', exist_ok=True)
os.makedirs('./training_stats/nocue', exist_ok=True)
os.makedirs('./training_stats/cue', exist_ok=True)


### Training
def net_train(net, optimizer, epoch, dataloader, lr_val=0.1):
    criterion = nn.CrossEntropyLoss()
    net.train()
    train_loss, correct, total = 0, 0, 0
    
    with tqdm(dataloader, unit=" batch") as tepoch:
        batch_idx = 0
        for inputs, targets in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}")
            batch_idx += 1
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            tepoch.set_postfix(loss=train_loss/batch_idx, accuracy=100. * correct/total, lr=lr_val)

    return [100. * correct/total]


### Main
if __name__ == "__main__":

    # setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default='VGG-13', choices=['VGG-13', 'res18'])
    parser.add_argument("--use_pretrained", type=int, default=-1)
    parser.add_argument("--base_dataset", default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'Dominoes'])
    parser.add_argument("--cue_type", default='nocue', choices=['nocue', 'box', 'dominoes'])
    parser.add_argument("--cue_proportion", type=float, default=1.)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_lr", type=float, default=0.1)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--lr_decay_epoch", type=int, default=40)
    parser.add_argument("--save_model", type=bool, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print("-- Model:", args.model)
    print("-- Dataset:", args.base_dataset)
    print("-- Using pretrained model from epoch", args.use_pretrained if args.use_pretrained > -1 else 'None')
    if (args.cue_type != 'nocue'):
        print("-- Cue type:", args.cue_type)
        print("-- Cue proportion:", args.cue_proportion)
    else:
        print("-- Training without cues")
    print("-- Training epoch:", args.n_epochs)
    print("-- LR decays at epoch {} with a factor of {}".format(args.lr_decay_epoch, args.lr_decay))
    print("-- Model saved after training:", args.save_model)
    print("\n")

    # dataloaders
    trainloader = syndata.get_dataloader(load_type='train', base_dataset=args.base_dataset, cue_type=args.cue_type, cue_proportion=args.cue_proportion, 
                                       batch_size=args.batch_size)

    nocue_testloader = syndata.get_dataloader(load_type='test', base_dataset=args.base_dataset, cue_type='nocue', batch_size=args.batch_size)

    if args.cue_type != 'nocue':
        cue_testloader = syndata.get_dataloader(load_type='test', base_dataset=args.base_dataset, cue_type=args.cue_type, cue_proportion=args.cue_proportion, 
                                            batch_size=args.batch_size)


    # model 
    net = create_model(args.model, num_classes=len(nocue_testloader.dataset.classes)).to(device)
    if(args.use_pretrained > -1):
        model_name = '{}_{}_net_nocue_{}_epochs_{}_pretrained'.format(args.base_dataset, args.model, args.n_epochs, args.use_pretrained if args.use_pretrained > -1 else 'not')
        PATH = './saved_models/nocue/'+model_name+'.pth'
        net.load_state_dict(torch.load(PATH))


    # train
    optimizer = optim.SGD(net.parameters(), lr=args.start_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=args.lr_decay)

    for epoch_lr in range(args.n_epochs):
        accs_dict['train'] += net_train(net, optimizer, epoch_lr, trainloader, lr_val=scheduler.get_last_lr()[0])
        accs_dict['test_nocue'] += linear_eval(net, dataloader=nocue_testloader, suffix='nocue')
        if args.cue_type != 'nocue':
            accs_dict['test_cue'] += linear_eval(net, dataloader=cue_testloader, suffix=args.cue_type)
        print("\n")
        scheduler.step()
        torch.cuda.empty_cache()


    # save model and training stats
    if args.save_model:
        if args.cue_type != 'nocue':
            model_name = '{}_{}_net_{}_epochs_{}_cue_{}_proportion_{}_pretrained_{}_seed'.format(
                        args.base_dataset, args.model, args.n_epochs, args.cue_type, args.cue_proportion, 
                        args.use_pretrained if args.use_pretrained > -1 else 'not', args.seed)
        else:
            model_name = '{}_{}_net_nocue_{}_epochs_{}_pretrained'.format(args.base_dataset, args.model, args.n_epochs, 
                        args.use_pretrained if args.use_pretrained > -1 else 'not')

        if(args.cue_type=='nocue'):
            torch.save(net.state_dict(), './saved_models/nocue/'+model_name+'.pth')
            with open('./training_stats/nocue/'+model_name+'.pkl', 'wb') as f:
                pkl.dump(accs_dict, f)
        else:
            torch.save(net.state_dict(), './saved_models/cue/'+model_name+'.pth')
            with open('./training_stats/cue/'+model_name+'.pkl', 'wb') as f:
                pkl.dump(accs_dict, f)
