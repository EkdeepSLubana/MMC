import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
torch.manual_seed(int(0))
cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
from scipy.optimize import linear_sum_assignment



###### VGG permutation identifier 
def VGG_permute(net_permuted, net_target, dataloader, n_match_iters=90, verbose=False):
    net_permuted.eval()
    net_target.eval()

    with torch.no_grad():

        # Conv / BN layers
        for lnum in range(len(net_permuted.layers)):
            for match_iter, (x, y) in enumerate(dataloader):
                if(match_iter > n_match_iters):
                    break
                x, y = x.to(device), y.to(device)

                # get activations at layer lnum            
                z_target = net_target.layers[0:lnum+1](x)
                z_permuted = net_permuted.layers[0:lnum+1](x)
                n_neurons = z_target.shape[1]

                # similarity matrix
                corr_matrix = F.cosine_similarity((z_target - z_target.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(1), 
                                                  (z_permuted - z_permuted.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(0), dim=-1)
                del z_target, z_permuted
                if verbose and (match_iter==0 or match_iter==n_match_iters-1):
                    scores, _ = corr_matrix.max(dim=1)
                    print("Average max similarity at iteration {}: {}".format(match_iter, scores.mean().item()))

                # matching
                _, assigns = linear_sum_assignment(-corr_matrix.cpu().numpy())

                # reassign weights
                net_permuted.layers[lnum].conv1.weight.data = net_permuted.layers[lnum].conv1.weight[assigns].clone()
                net_permuted.layers[lnum].bn1.weight.data = net_permuted.layers[lnum].bn1.weight[assigns].clone()
                net_permuted.layers[lnum].bn1.bias.data = net_permuted.layers[lnum].bn1.bias[assigns].clone()
                net_permuted.layers[lnum].bn1.running_mean.data = net_permuted.layers[lnum].bn1.running_mean[assigns].clone()
                net_permuted.layers[lnum].bn1.running_var.data = net_permuted.layers[lnum].bn1.running_var[assigns].clone()

                if(lnum < len(net_permuted.layers)-1):
                    net_permuted.layers[lnum+1].conv1.weight.data = net_permuted.layers[lnum+1].conv1.weight[:,assigns,:,:].clone()
                elif(lnum == len(net_permuted.layers)-1):
                    net_permuted.linear.weight.data = net_permuted.linear.weight[:,assigns].clone()
                    net_permuted.linear.bias.data = net_permuted.linear.bias.clone()

                torch.cuda.empty_cache()

    return net_permuted



###### ResNet permutation identifier
def res_permute(net_permuted, net_target, dataloader, n_match_iters=90, verbose=False):
    net_permuted.eval()
    net_target.eval()
    
    with torch.no_grad():

        ###### Conv1 / BN1 layers
        for match_iter, (x, y) in enumerate(dataloader):
            # print(match_iter)
            if(match_iter > n_match_iters):
                break
            x, y = x.to(device), y.to(device)

            # get activations
            z_target = F.relu(net_target.bn1(net_target.conv1(x)))
            z_permuted = F.relu(net_permuted.bn1(net_permuted.conv1(x)))
            n_samples, n_neurons, n_dim = z_target.shape[0], z_target.shape[1], z_target.shape[2]

            # similarity matrix
            corr_matrix = F.cosine_similarity((z_target - z_target.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(1), 
                                                  (z_permuted - z_permuted.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(0), dim=-1)
            if verbose and (match_iter==0 or match_iter==n_match_iters-1):
                scores, _ = corr_matrix.max(dim=1)
                print("Average max similarity at iteration {}: {}".format(match_iter, scores.mean().item()))
            del z_target, z_permuted

            # matching
            _, assigns_in = linear_sum_assignment(-corr_matrix.cpu().numpy())

            # reassign weights
            net_permuted.conv1.weight.data = net_permuted.conv1.weight.data[assigns_in].clone()
            net_permuted.bn1.weight.data = net_permuted.bn1.weight[assigns_in].clone()
            net_permuted.bn1.bias.data = net_permuted.bn1.bias[assigns_in].clone()
            net_permuted.bn1.running_mean.data = net_permuted.bn1.running_mean[assigns_in].clone()
            net_permuted.bn1.running_var.data = net_permuted.bn1.running_var[assigns_in].clone()

            torch.cuda.empty_cache()
        
            ###### Module 1
            # print("\nModule 1")
            for lnum in range(2):
                
                ### get base activations
                z_target = F.relu(net_target.bn1(net_target.conv1(x)))
                z_permuted = F.relu(net_permuted.bn1(net_permuted.conv1(x)))

                ### get activations at conv 1 / bn 1
                z_target = F.relu(net_target.layer1[lnum].bn1(net_target.layer1[lnum].conv1(z_target)))
                z_permuted = F.relu(net_permuted.layer1[lnum].bn1(net_permuted.layer1[lnum].conv1(z_permuted)))
                n_samples, n_neurons, n_dim = z_target.shape[0], z_target.shape[1], z_target.shape[2]

                # similarity matrix
                corr_matrix = F.cosine_similarity((z_target - z_target.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(1), 
                                                    (z_permuted - z_permuted.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(0), dim=-1)
                if verbose and (match_iter==0 or match_iter==n_match_iters-1):
                    scores, _ = corr_matrix.max(dim=1)
                    print("Average max similarity at iteration {}: {}".format(match_iter, scores.mean().item()))
                
                # matching
                _, assigns = linear_sum_assignment(-corr_matrix.cpu().numpy())

                # reassign weights
                net_permuted.layer1[lnum].conv1.weight.data = net_permuted.layer1[lnum].conv1.weight[assigns][:, assigns_in, :, :].clone()
                net_permuted.layer1[lnum].bn1.weight.data = net_permuted.layer1[lnum].bn1.weight[assigns].clone()
                net_permuted.layer1[lnum].bn1.bias.data = net_permuted.layer1[lnum].bn1.bias[assigns].clone()
                net_permuted.layer1[lnum].bn1.running_mean.data = net_permuted.layer1[lnum].bn1.running_mean[assigns].clone()
                net_permuted.layer1[lnum].bn1.running_var.data = net_permuted.layer1[lnum].bn1.running_var[assigns].clone()

                ### conv 2 / bn 2
                net_permuted.layer1[lnum].conv2.weight.data = net_permuted.layer1[lnum].conv2.weight[assigns_in][:, assigns, :, :].clone()
                net_permuted.layer1[lnum].bn2.weight.data = net_permuted.layer1[lnum].bn2.weight[assigns_in].clone()
                net_permuted.layer1[lnum].bn2.bias.data = net_permuted.layer1[lnum].bn2.bias[assigns_in].clone()
                net_permuted.layer1[lnum].bn2.running_mean.data = net_permuted.layer1[lnum].bn2.running_mean[assigns_in].clone()
                net_permuted.layer1[lnum].bn2.running_var.data = net_permuted.layer1[lnum].bn2.running_var[assigns_in].clone()
                torch.cuda.empty_cache()

                del z_target, z_permuted


            ###### Module 2
            # print("\nModule 2")
            for lnum in range(2): 

                ### get base activations
                if lnum == 0:
                    z_target = net_target.layer1(F.relu(net_target.bn1(net_target.conv1(x))))
                    z_permuted = net_permuted.layer1(F.relu(net_permuted.bn1(net_permuted.conv1(x))))
                else:
                    z_target = net_target.layer2[0](net_target.layer1(F.relu(net_target.bn1(net_target.conv1(x)))))
                    z_permuted = net_permuted.layer2[0](net_permuted.layer1(F.relu(net_permuted.bn1(net_permuted.conv1(x)))))


                ### get activations at conv 1 / bn 1
                z_target1 = F.relu(net_target.layer2[lnum].bn1(net_target.layer2[lnum].conv1(z_target)))
                z_permuted1 = F.relu(net_permuted.layer2[lnum].bn1(net_permuted.layer2[lnum].conv1(z_permuted)))
                n_samples, n_neurons, n_dim = z_target1.shape[0], z_target1.shape[1], z_target1.shape[2]

                # similarity matrix
                corr_matrix = F.cosine_similarity((z_target1 - z_target1.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(1), 
                                                  (z_permuted1 - z_permuted1.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(0), dim=-1)
                if verbose and (match_iter==0 or match_iter==n_match_iters-1):
                    scores, _ = corr_matrix.max(dim=1)
                    print("Average max similarity at iteration {}: {}".format(match_iter, scores.mean().item()))
                
                # matching
                _, assigns = linear_sum_assignment(-corr_matrix.cpu().numpy())

                # reassign weights
                net_permuted.layer2[lnum].conv1.weight.data = net_permuted.layer2[lnum].conv1.weight[assigns][:, assigns_in, :, :].clone()
                net_permuted.layer2[lnum].bn1.weight.data = net_permuted.layer2[lnum].bn1.weight[assigns].clone()
                net_permuted.layer2[lnum].bn1.bias.data = net_permuted.layer2[lnum].bn1.bias[assigns].clone()
                net_permuted.layer2[lnum].bn1.running_mean.data = net_permuted.layer2[lnum].bn1.running_mean[assigns].clone()
                net_permuted.layer2[lnum].bn1.running_var.data = net_permuted.layer2[lnum].bn1.running_var[assigns].clone()
                torch.cuda.empty_cache()
                

                ### get activations at layer 2's shortcut
                if(lnum == 0):
                    z_target_s = F.relu(net_target.layer2[lnum].shortcut_bn(net_target.layer2[lnum].shortcut_conv(z_target)))
                    z_permuted_s = F.relu(net_permuted.layer2[lnum].shortcut_bn(net_permuted.layer2[lnum].shortcut_conv(z_permuted)))
                    n_samples, n_neurons, n_dim = z_target_s.shape[0], z_target_s.shape[1], z_target_s.shape[2]

                    # similarity matrix
                    corr_matrix = F.cosine_similarity((z_target_s - z_target_s.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(1), 
                                                  (z_permuted_s - z_permuted_s.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(0), dim=-1)
                    if verbose and (match_iter==0 or match_iter==n_match_iters-1):
                        scores, _ = corr_matrix.max(dim=1)
                        print("Average max similarity at iteration {}: {}".format(match_iter, scores.mean().item()))
                    del z_target_s, z_permuted_s

                    # matching
                    _, assigns_s = linear_sum_assignment(-corr_matrix.cpu().numpy())

                    # reassign weights
                    net_permuted.layer2[lnum].shortcut_conv.weight.data = net_permuted.layer2[lnum].shortcut_conv.weight[assigns_s][:,assigns_in,:,:].clone()
                    net_permuted.layer2[lnum].shortcut_bn.weight.data = net_permuted.layer2[lnum].shortcut_bn.weight[assigns_s].clone()
                    net_permuted.layer2[lnum].shortcut_bn.bias.data = net_permuted.layer2[lnum].shortcut_bn.bias[assigns_s].clone()
                    net_permuted.layer2[lnum].shortcut_bn.running_mean.data = net_permuted.layer2[lnum].shortcut_bn.running_mean[assigns_s].clone()
                    net_permuted.layer2[lnum].shortcut_bn.running_var.data = net_permuted.layer2[lnum].shortcut_bn.running_var[assigns_s].clone()
                    assigns_in = assigns_s.copy()
                    torch.cuda.empty_cache()


                ### get activations at conv 2 / bn 2
                net_permuted.layer2[lnum].conv2.weight.data = net_permuted.layer2[lnum].conv2.weight[assigns_in][:, assigns, :, :].clone()
                net_permuted.layer2[lnum].bn2.weight.data = net_permuted.layer2[lnum].bn2.weight[assigns_in].clone()
                net_permuted.layer2[lnum].bn2.bias.data = net_permuted.layer2[lnum].bn2.bias[assigns_in].clone()
                net_permuted.layer2[lnum].bn2.running_mean.data = net_permuted.layer2[lnum].bn2.running_mean[assigns_in].clone()
                net_permuted.layer2[lnum].bn2.running_var.data = net_permuted.layer2[lnum].bn2.running_var[assigns_in].clone()
                torch.cuda.empty_cache()


            ###### Module 3
            # print("\nModule 3")
            for lnum in range(2):
                
                ### get base activations
                if lnum == 0:
                    z_target = net_target.layer2(net_target.layer1(F.relu(net_target.bn1(net_target.conv1(x)))))
                    z_permuted = net_permuted.layer2(net_permuted.layer1(F.relu(net_permuted.bn1(net_permuted.conv1(x)))))
                else:
                    z_target = net_target.layer3[0](net_target.layer2(net_target.layer1(F.relu(net_target.bn1(net_target.conv1(x))))))
                    z_permuted = net_permuted.layer3[0](net_permuted.layer2(net_permuted.layer1(F.relu(net_permuted.bn1(net_permuted.conv1(x))))))


                ### get activations at conv 1 / bn 1
                z_target1 = F.relu(net_target.layer3[lnum].bn1(net_target.layer3[lnum].conv1(z_target)))
                z_permuted1 = F.relu(net_permuted.layer3[lnum].bn1(net_permuted.layer3[lnum].conv1(z_permuted)))
                n_samples, n_neurons, n_dim = z_target1.shape[0], z_target1.shape[1], z_target1.shape[2]

                # similarity matrix
                corr_matrix = F.cosine_similarity((z_target1 - z_target1.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(1), 
                                                  (z_permuted1 - z_permuted1.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(0), dim=-1)
                if verbose and (match_iter==0 or match_iter==n_match_iters-1):
                    scores, _ = corr_matrix.max(dim=1)
                    print("Average max similarity at iteration {}: {}".format(match_iter, scores.mean().item()))
                
                # matching
                _, assigns = linear_sum_assignment(-corr_matrix.cpu().numpy())

                # reassign weights
                net_permuted.layer3[lnum].conv1.weight.data = net_permuted.layer3[lnum].conv1.weight[assigns][:, assigns_in, :, :].clone()
                net_permuted.layer3[lnum].bn1.weight.data = net_permuted.layer3[lnum].bn1.weight[assigns].clone()
                net_permuted.layer3[lnum].bn1.bias.data = net_permuted.layer3[lnum].bn1.bias[assigns].clone()
                net_permuted.layer3[lnum].bn1.running_mean.data = net_permuted.layer3[lnum].bn1.running_mean[assigns].clone()
                net_permuted.layer3[lnum].bn1.running_var.data = net_permuted.layer3[lnum].bn1.running_var[assigns].clone()
                torch.cuda.empty_cache()


                ### get activations at layer 3's shortcut
                if(lnum == 0):
                    z_target_s = F.relu(net_target.layer3[lnum].shortcut_bn(net_target.layer3[lnum].shortcut_conv(z_target)))
                    z_permuted_s = F.relu(net_permuted.layer3[lnum].shortcut_bn(net_permuted.layer3[lnum].shortcut_conv(z_permuted)))
                    n_samples, n_neurons, n_dim = z_target_s.shape[0], z_target_s.shape[1], z_target_s.shape[2]

                    # similarity matrix
                    corr_matrix = F.cosine_similarity((z_target_s - z_target_s.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(1), 
                                                  (z_permuted_s - z_permuted_s.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(0), dim=-1)
                    if verbose and (match_iter==0 or match_iter==n_match_iters-1):
                        scores, _ = corr_matrix.max(dim=1)
                        print("Average max similarity at iteration {}: {}".format(match_iter, scores.mean().item()))
                    del z_target_s, z_permuted_s

                    # matching
                    _, assigns_s = linear_sum_assignment(-corr_matrix.cpu().numpy())

                    # reassign weights
                    net_permuted.layer3[lnum].shortcut_conv.weight.data = net_permuted.layer3[lnum].shortcut_conv.weight[assigns_s][:,assigns_in,:,:].clone()
                    net_permuted.layer3[lnum].shortcut_bn.weight.data = net_permuted.layer3[lnum].shortcut_bn.weight[assigns_s].clone()
                    net_permuted.layer3[lnum].shortcut_bn.bias.data = net_permuted.layer3[lnum].shortcut_bn.bias[assigns_s].clone()
                    net_permuted.layer3[lnum].shortcut_bn.running_mean.data = net_permuted.layer3[lnum].shortcut_bn.running_mean[assigns_s].clone()
                    net_permuted.layer3[lnum].shortcut_bn.running_var.data = net_permuted.layer3[lnum].shortcut_bn.running_var[assigns_s].clone()
                    assigns_in = assigns_s.copy()
                    torch.cuda.empty_cache()

                    
                ### get activations at conv 2 / bn 2
                net_permuted.layer3[lnum].conv2.weight.data = net_permuted.layer3[lnum].conv2.weight[assigns_in][:, assigns, :, :].clone()
                net_permuted.layer3[lnum].bn2.weight.data = net_permuted.layer3[lnum].bn2.weight[assigns_in].clone()
                net_permuted.layer3[lnum].bn2.bias.data = net_permuted.layer3[lnum].bn2.bias[assigns_in].clone()
                net_permuted.layer3[lnum].bn2.running_mean.data = net_permuted.layer3[lnum].bn2.running_mean[assigns_in].clone()
                net_permuted.layer3[lnum].bn2.running_var.data = net_permuted.layer3[lnum].bn2.running_var[assigns_in].clone()

                torch.cuda.empty_cache()
                

            ###### Module 4
            # print("\nModule 4")
            for lnum in range(2):
                
                ### get base activations
                if lnum == 0:
                    z_target = net_target.layer3(net_target.layer2(net_target.layer1(F.relu(net_target.bn1(net_target.conv1(x))))))
                    z_permuted = net_permuted.layer3(net_permuted.layer2(net_permuted.layer1(F.relu(net_permuted.bn1(net_permuted.conv1(x))))))
                else:
                    z_target = net_target.layer4[0](net_target.layer3(net_target.layer2(net_target.layer1(F.relu(net_target.bn1(net_target.conv1(x)))))))
                    z_permuted = net_permuted.layer4[0](net_permuted.layer3(net_permuted.layer2(net_permuted.layer1(F.relu(net_permuted.bn1(net_permuted.conv1(x)))))))


                ### get activations at conv 1 / bn 1
                z_target1 = F.relu(net_target.layer4[lnum].bn1(net_target.layer4[lnum].conv1(z_target)))
                z_permuted1 = F.relu(net_permuted.layer4[lnum].bn1(net_permuted.layer4[lnum].conv1(z_permuted)))
                n_samples, n_neurons, n_dim = z_target1.shape[0], z_target1.shape[1], z_target1.shape[2]

                # similarity matrix
                corr_matrix = F.cosine_similarity((z_target1 - z_target1.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(1), 
                                                  (z_permuted1 - z_permuted1.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(0), dim=-1)
                if verbose and (match_iter==0 or match_iter==n_match_iters-1):
                    scores, _ = corr_matrix.max(dim=1)
                    print("Average max similarity at iteration {}: {}".format(match_iter, scores.mean().item()))
                
                # matching
                _, assigns = linear_sum_assignment(-corr_matrix.cpu().numpy())

                # reassign weights
                net_permuted.layer4[lnum].conv1.weight.data = net_permuted.layer4[lnum].conv1.weight[assigns][:, assigns_in, :, :].clone()
                net_permuted.layer4[lnum].bn1.weight.data = net_permuted.layer4[lnum].bn1.weight[assigns].clone()
                net_permuted.layer4[lnum].bn1.bias.data = net_permuted.layer4[lnum].bn1.bias[assigns].clone()
                net_permuted.layer4[lnum].bn1.running_mean.data = net_permuted.layer4[lnum].bn1.running_mean[assigns].clone()
                net_permuted.layer4[lnum].bn1.running_var.data = net_permuted.layer4[lnum].bn1.running_var[assigns].clone()

                torch.cuda.empty_cache()
                
                
                ### get activations at layer 2's shortcut
                if(lnum == 0):
                    z_target_s = F.relu(net_target.layer4[lnum].shortcut_bn(net_target.layer4[lnum].shortcut_conv(z_target)))
                    z_permuted_s = F.relu(net_permuted.layer4[lnum].shortcut_bn(net_permuted.layer4[lnum].shortcut_conv(z_permuted)))
                    n_samples, n_neurons, n_dim = z_target_s.shape[0], z_target_s.shape[1], z_target_s.shape[2]

                    # similarity matrix
                    corr_matrix = F.cosine_similarity((z_target_s - z_target_s.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(1), 
                                                  (z_permuted_s - z_permuted_s.mean(dim=(0, 2, 3), keepdim=True)).permute(1, 2, 3, 0).reshape(n_neurons, -1).unsqueeze(0), dim=-1)
                    if verbose and (match_iter==0 or match_iter==n_match_iters-1):
                        scores, _ = corr_matrix.max(dim=1)
                        print("Average max similarity at iteration {}: {}".format(match_iter, scores.mean().item()))
                    del z_target_s, z_permuted_s

                    # matching
                    _, assigns_s = linear_sum_assignment(-corr_matrix.cpu().numpy())

                    # reassign weights
                    net_permuted.layer4[lnum].shortcut_conv.weight.data = net_permuted.layer4[lnum].shortcut_conv.weight[assigns_s][:,assigns_in,:,:].clone()
                    net_permuted.layer4[lnum].shortcut_bn.weight.data = net_permuted.layer4[lnum].shortcut_bn.weight[assigns_s].clone()
                    net_permuted.layer4[lnum].shortcut_bn.bias.data = net_permuted.layer4[lnum].shortcut_bn.bias[assigns_s].clone()
                    net_permuted.layer4[lnum].shortcut_bn.running_mean.data = net_permuted.layer4[lnum].shortcut_bn.running_mean[assigns_s].clone()
                    net_permuted.layer4[lnum].shortcut_bn.running_var.data = net_permuted.layer4[lnum].shortcut_bn.running_var[assigns_s].clone()
                    assigns_in = assigns_s.copy()
                    torch.cuda.empty_cache()


                ### get activations at conv 2 / bn 2
                net_permuted.layer4[lnum].conv2.weight.data = net_permuted.layer4[lnum].conv2.weight[assigns_in][:, assigns, :, :].clone()
                net_permuted.layer4[lnum].bn2.weight.data = net_permuted.layer4[lnum].bn2.weight[assigns_in].clone()
                net_permuted.layer4[lnum].bn2.bias.data = net_permuted.layer4[lnum].bn2.bias[assigns_in].clone()
                net_permuted.layer4[lnum].bn2.running_mean.data = net_permuted.layer4[lnum].bn2.running_mean[assigns_in].clone()
                net_permuted.layer4[lnum].bn2.running_var.data = net_permuted.layer4[lnum].bn2.running_var[assigns_in].clone()


            net_permuted.linear.weight.data = net_permuted.linear.weight[:,assigns_in].clone()
            net_permuted.linear.bias.data = net_permuted.linear.bias.clone()


    return net_permuted