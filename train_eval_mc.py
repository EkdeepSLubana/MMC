import torch
import torch.backends.cudnn as cudnn
import argparse
import os
import pickle as pkl
from syndata import get_dataloader
from utils import change_setup
from models import create_model
import mode_connect
from find_permutation import VGG_permute, res_permute

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Dir for saving midpoint models
os.makedirs('./midpoints', exist_ok=True)
for ddir in ['CIFAR10', 'CIFAR100', 'Dominoes']:
    os.makedirs('./midpoints/'+ddir+'/cue', exist_ok=True)

### Dir for saving eval results
os.makedirs('./results', exist_ok=True)
for dset in ['CIFAR10', 'CIFAR100', 'Dominoes']:
    os.makedirs('./results/'+dset, exist_ok=True)
    for ddir in ['train', 'test']:
        os.makedirs('./results/'+dset+'/'+ddir, exist_ok=True)
        os.makedirs('./results/'+dset+'/'+ddir+'/cue', exist_ok=True)

### Main
if __name__ == "__main__":

    ## setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perform", default='eval_path', choices=['train_midpoint', 'eval_path'])
    parser.add_argument("--model", default='VGG-13', choices=['VGG-13', 'res18'])
    parser.add_argument("--base_dataset", default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'Dominoes'])
    parser.add_argument("--connectivity_pattern", default='QMC', choices=['LMC', 'LMCP', 'QMC'])
    parser.add_argument("--cue_type", default='box', choices=['nocue', 'box', 'dominoes'])
    parser.add_argument("--cue_proportion", default=1.0, type=float, choices=[0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--start_lr", type=float, default=0.1)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--decay_epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default='SGD')
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--path_cue", type=str, default=None)
    parser.add_argument("--path_nocue", type=str, default=None)
    parser.add_argument("--id_data", choices=['nocue', 'cue'])
    parser.add_argument("--eval_data", default='train', choices=['train', 'test'])
    parser.add_argument("--n_interpolations", type=int, default=5)
    parser.add_argument("--n_match_iters", type=int, default=90)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    save_path = args.save_path
    if(args.path_cue == None):
        path_cue = './saved_models/{}_{}_net_100_epochs_{}_cue_{:.1f}_proportion_1.0_not_pretrained_{}_seed.pth'.format(
            args.base_dataset, args.model, args.cue_type, args.cue_proportion, args.seed)
    else:
        path_cue = args.path_cue

    if(args.path_nocue == None):
        path_nocue = './saved_models/nocue/{}_{}_net_nocue_100_epochs_not_pretrained.pth'.format(args.base_dataset, args.model)
    else:
        path_nocue = args.path_nocue

    if args.perform == 'train_midpoint':
        if args.save_path == None:
            save_path = './midpoints/{}/{}_proportion_{:.1f}_idset_{}_nepochs_{}_seed_{}.pth'.format(args.base_dataset, 
                                args.model, args.cue_proportion, args.id_data, args.n_epochs, args.seed)
        else:
            raise Exception("Save path should not be defined for midpoint training.")

    elif args.perform == 'eval_path':
        if(args.path_cue == None):
            path_midpoint = './midpoints/{}/{}_proportion_{:.1f}_idset_{}_nepochs_{}_seed_{}.pth'.format(args.base_dataset, 
                    args.model, args.cue_proportion, args.id_data, args.n_epochs, args.seed)

    if args.save_path == None:
        if args.connectivity_pattern == 'LMC':
            save_path = './results/{}/{}/{}_lmc_proportion_{:.1f}_seed_{}'.format(args.base_dataset, args.eval_data,
                        args.model, args.cue_proportion, args.seed)
        if args.save_path == None:
            if args.connectivity_pattern == 'LMCP':
                save_path = './results/{}/{}/{}_lmcp_proportion_{:.1f}_seed_{}'.format(args.base_dataset, args.eval_data,
                            args.model, args.cue_proportion, args.seed)
            
            elif args.connectivity_pattern == 'QMC':
                save_path = './results/{}/{}/{}_qmc_proportion_{:.1f}_idset_{}_nepochs_{}_seed_{}'.format(args.base_dataset, args.eval_data,
                            args.model, args.cue_proportion, args.id_data, args.n_epochs, args.seed)

    # print setup
    torch.manual_seed(args.seed)
    print("-- Model:", args.model)
    print("-- Base Dataset:", args.base_dataset)
    print("-- Eval / ID via:", args.eval_data if args.perform == 'eval_path' else args.id_data)
    print("-- Cue model comes from:", path_cue)
    print("-- No-Cue model comes from:", path_nocue)
    print("-- Connectivity pattern", args.connectivity_pattern)
    print("-- Cue type:", args.cue_type)
    print("-- Cue proportion:", args.cue_proportion)
    print("-- Epochs to train for:", args.n_epochs)
    print("-- LR decays every {} epochs with a factor of {}".format(args.decay_epochs, args.lr_decay))
    print("-- Model saved after training:", args.save_model)
    if(args.save_model):
        print("-- Saved to:", args.save_path)
    print("\n")

    # minor kinks
    nocue_testloader = get_dataloader(load_type='test', base_dataset=args.base_dataset, cue_type='nocue')
    args.num_classes = len(nocue_testloader.dataset.classes)
    del nocue_testloader


    ## define cue and no-cue models
    net_c = create_model(args.model, num_classes=args.num_classes).to(device)
    net_nc = create_model(args.model, num_classes=args.num_classes).to(device)


    ## load models
    net_c.load_state_dict(torch.load(path_cue))
    net_nc.load_state_dict(torch.load(path_nocue))


    ###### Midpoint Identification 
    if(args.perform == 'train_midpoint'):
        net_midpoint, accs_midpoint = mode_connect.train_quad_midpoint(net_1=net_c, net_2=net_nc, train_setup=vars(args))

        # save midpoint model
        torch.save(net_midpoint.state_dict(), save_path)


    ###### Counterfactual Evaluation of a Path
    elif(args.perform == 'eval_path'):

        # permute, if desired
        if args.connectivity_pattern == 'LMCP':
            print("Finding permutation to match models... ", end="")
            dataloader = get_dataloader(load_type='train', base_dataset=args.base_dataset, cue_type='nocue')
            if args.model == 'VGG-13':
                net_c = VGG_permute(net_permuted=net_c, net_target=net_nc, dataloader=dataloader, n_match_iters=args.n_match_iters)
            elif args.model == 'res18':
                net_c = res_permute(net_permuted=net_c, net_target=net_nc, dataloader=dataloader, n_match_iters=args.n_match_iters)
            print("done.")

        # load midpoint model for QMC
        net_midpoint = None
        if(args.connectivity_pattern == 'QMC'):
            net_midpoint = create_model(args.model, num_classes=args.num_classes).to(device)
            net_midpoint.load_state_dict(torch.load(path_midpoint))

        # defining this dict explicitly makes things easier--let's stick with this
        eval_setup = {
                  'model_class': args.model,
                  'base_dataset': args.base_dataset,
                  'num_classes': args.num_classes,
                  'n_interpolations': args.n_interpolations,
                  'eval_data': args.eval_data,
                  'cue_type': args.cue_type,
                  'randomize_cue': False,
                  'randomize_img': False,
                  'connect_pattern': args.connectivity_pattern,
                  'seed_id': args.seed
             }

        # run evaluation
        accs_dict, loss_dict = {}, {}
        if args.base_dataset == 'Dominoes':
            for (attr, val) in [('cue_type', 'nocue'), ('cue_type', 'dominoes'), ('randomize_cue', True), ('randomize_img', True)]:
                print("{}: {}".format(attr, val))
                setup = change_setup(eval_setup, attr_change=[attr], attr_val=[val])
                a, l = mode_connect.probe_connect(net_1=net_c.eval(), net_2=net_nc.eval(), net_midpoint=None if net_midpoint==None else net_midpoint.eval(), 
                                                    setup=setup, return_loss=True)
                accs_dict.update({(attr, val): a})
                loss_dict.update({(attr, val): l})

        else:
            for (attr, val) in [('cue_type', 'nocue'), ('cue_type', 'box'), ('randomize_cue', True), ('randomize_img', True)]:
                print("{}: {}".format(attr, val))
                setup = change_setup(eval_setup, attr_change=[attr], attr_val=[val])
                a, l = mode_connect.probe_connect(net_1=net_c.eval(), net_2=net_nc.eval(), net_midpoint=None if net_midpoint==None else net_midpoint.eval(), 
                                                    setup=setup, return_loss=True)
                accs_dict.update({(attr, val): a})
                loss_dict.update({(attr, val): l})


        ### Save results
        with open(save_path+'_accs.pkl', 'wb') as f:
            pkl.dump(accs_dict, f)

        with open(save_path+'_loss.pkl', 'wb') as f:
            pkl.dump(loss_dict, f)
