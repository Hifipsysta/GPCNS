import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
import psutil
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse,time
import math
from copy import deepcopy
from layers import Conv2d, Linear
## Define LeNet model 
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class LeNet(nn.Module):
    def __init__(self,taskcla):
        super(LeNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        
        self.map.append(32)
        self.conv1 = Conv2d(3, 20, 5, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(3)        
        self.map.append(s)
        self.conv2 = Conv2d(20, 50, 5, bias=False, padding=2)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(20)        
        self.smid=s
        self.map.append(50*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(3,2,padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)
        self.lrn = torch.nn.LocalResponseNorm(4,0.001/9.0,0.75,1)

        self.fc1 = nn.Linear(50*self.smid*self.smid,800, bias=False)
        self.fc2 = nn.Linear(800,500, bias=False)
        self.map.extend([800])
        
        self.taskcla = taskcla
        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(500,n,bias=False))
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))

        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.lrn (self.relu(x))))

        x=x.reshape(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(x))

        y=[]
        for t,i in self.taskcla:
            y.append(self.fc3[t](x))
            
        return y

def init_weights(m):
    # print(m)
    if isinstance(m, Linear) or isinstance(m, Conv2d) or isinstance(m, nn.Linear):
    # if type(m) == Linear or type(m) == Conv2d:
        print(m)
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return
def save_model(model, memory, savename):
    ckpt = {
        'model': model.state_dict(),
        'memory': memory,
    }

    # Save to file.
    torch.save(ckpt, savename+'checkpoint.pt')
    print(savename)
    return 
def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor  

def train(args, model, device, x,y, optimizer,criterion, task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        b = b.cpu()
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output[task_id], target)        
        loss.backward()
        optimizer.step()

def train_projected_regime (args, model,device,x,y,optimizer,criterion, task_name, task_name_list, grad_list, task_id, projection_list):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        b = b.cpu()
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output[task_id], target)         
        loss.backward()        
        
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):

            if 'weight' in m:
                if k<8 and len(params.size())!=1:
                    sz =  params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                            projection_list[kk]).view(params.size())
                    kk +=1
                elif (k<8 and len(params.size())==1) and task_id !=0 :
                    params.grad.data.fill_(0)
        

        optimizer.step()

            

def test(args, model, device, x, y, criterion, task_id):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            b = b.cpu()
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def get_gradient_each_layer (net, device, optimizer, criterion, task_id, x, y=None): 
    # Collect activations by forward pass
    grad_list = []

    k_linear = 0
    for k, (m,params) in enumerate(net.named_parameters()):
        if 'weight' in m and 'bn' not in m:
            if len(params.shape) == 4:
                
                grad = params.grad.data
                grad = grad.reshape(grad.shape[0], grad.shape[1]*grad.shape[2]*grad.shape[3])
                grad_list.append(grad)
            else:
                if 'fc3' in m and k_linear == task_id:
                    grad = params.grad.data
                    grad_list.append(grad)
                    k_linear += 1
                elif 'fc3' not in m:
                    grad = params.grad.data
                    grad_list.append(grad)  

    print('-'*30)
    print('Gradient Matrix')
    print('-'*30)
    for i in range(len(grad_list)):
        print ('Layer {} : {}'.format(i+1,grad_list[i].shape))
        # print ('Layer {} : {}'.format(i+1,grad_list[i].shape))

    print('-'*30)

    return grad_list



def get_space_and_grad(model, device, grad_list, threshold, task_name, task_name_list, task_id, Gradient_alltask_list, Nullspace_common_list, importance_list, Nullspace_alltask_list):
    print_log ('Threshold:{}'.format(threshold), log) 
    Ours = True
    if task_name == 'cifar100-0':
        # After First Task 
        for i in range(len(grad_list)):
            Gradient_matrix = grad_list[i]
            # print('Gradient Matrix', Gradient_matrix.shape)
            Gradient_alltask_list.append(Gradient_matrix)

            U_matrix, Sigma, Vh_matrix = torch.linalg.svd(Gradient_alltask_list[i], full_matrices=False)
            V_matrix = Vh_matrix.transpose(0,1)
            
            Sigma_total = torch.sum(Sigma)
            Sigma_accumul_list = [torch.sum(Sigma[0:idx+1]) for idx in range(len(Sigma))]
            Sigma_select = torch.tensor(Sigma_accumul_list).to(device) <= threshold[i]*Sigma_total
            Sigma_slim = Sigma[Sigma_select]
            rank_subspace = torch.count_nonzero(Sigma_slim)

            V_tilde_matrix = V_matrix[:, :rank_subspace]

            Nullspace_common_list.append(V_tilde_matrix)
            Nullspace_alltask_list.append(V_tilde_matrix)
            # print('Nullspace_common_list[i]', Nullspace_common_list[i].shape)

            importance_list.append(torch.ones(Nullspace_common_list[i].shape[1]).to(device))

    else:
        for i in range(len(grad_list)):
            Gradient_matrix = grad_list[i]
            # print('Gradient Matrix', Gradient_matrix.shape)
            Gradient_alltask_list[i] = torch.vstack([Gradient_alltask_list[i], Gradient_matrix])   
            # print('Matrix waiting to be decomposed', Gradient_alltask_list[i].shape)

            U_matrix, Sigma, Vh_matrix = torch.linalg.svd(Gradient_alltask_list[i], full_matrices=False) 
            V_matrix = Vh_matrix.transpose(0,1)

            Sigma_total = torch.sum(Sigma)
            Sigma_accumul_list = [torch.sum(Sigma[0:idx+1]).item() for idx in range(len(Sigma))]
            Sigma_select = torch.tensor(Sigma_accumul_list).to(device) <= threshold[i]*Sigma_total
            Sigma_slim = Sigma[Sigma_select]
            rank_subspace = torch.count_nonzero(Sigma_slim)
            # print('Rank of Matrix：', rank_subspace)

            V_tilde_matrix = V_matrix[:, :rank_subspace]

            
            Nullspace_alltask_list[i] = torch.hstack([Nullspace_alltask_list[i], V_tilde_matrix])
            U_matrix, Sigma, Vh_matrix = torch.linalg.svd(Nullspace_alltask_list[i], full_matrices=False)


            Sigma_total = torch.sum(Sigma)
            Sigma_accumul_list = [torch.sum(Sigma[0:idx+1]).item() for idx in range(len(Sigma))]
            Sigma_select = torch.tensor(Sigma_accumul_list).to(device) <= threshold[i]*Sigma_total
            Sigma_slim = Sigma[Sigma_select]
            rank_subspace = torch.count_nonzero(Sigma_slim)

            V_tilde_matrix = U_matrix[:, :rank_subspace]
            # print('V_tilde_matrix', V_tilde_matrix.shape)

            importance = ((args.scale_coff+1)*Sigma[:rank_subspace])/(args.scale_coff*Sigma[:rank_subspace] + max(Sigma[:rank_subspace]))         

            importance_list[i] = importance
            # print(importance_list[i])

            Nullspace_common_list[i] = V_tilde_matrix
            # print('Nullspace_common_list[i]', Nullspace_common_list[i].shape)


    print_log('-'*40, log)
    print_log('Common Null Space', log)
    print_log('-'*40, log)

    for i in range(4):
        print_log ('Layer {} : {}/{}'.format(i+1,Nullspace_common_list[i].shape[1], Nullspace_common_list[i].shape[0]), log)
    print_log('-'*40, log)
    
    return Nullspace_common_list, importance_list, Gradient_alltask_list, Nullspace_alltask_list





def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Choose any task order - ref {yoon et al. ICLR 2020}
    task_order = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                  np.array([15, 12, 5, 9, 7, 16, 18, 17, 1, 0, 3, 8, 11, 14, 10, 6, 2, 4, 13, 19]),
                  np.array([17, 1, 19, 18, 12, 7, 6, 0, 11, 15, 10, 5, 13, 3, 9, 16, 4, 14, 2, 8]),
                  np.array([11, 9, 6, 5, 12, 4, 0, 10, 13, 7, 14, 3, 15, 16, 8, 1, 2, 19, 18, 17]),
                  np.array([6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16])]

    ## Load CIFAR100_SUPERCLASS DATASET
    from dataloader import cifar100_superclass as data_loader
    data, taskcla = data_loader.cifar100_superclass_python(task_order[args.t_order], group=5, validation=True)
    test_data,_   = data_loader.cifar100_superclass_python(task_order[args.t_order], group=5)
    print (taskcla)

    acc_matrix=np.zeros((20,20))
    criterion = torch.nn.CrossEntropyLoss()
    task_id = 0
    task_list = []
    task_name_list = []
    memory = {}

    Gradient_alltask_list = []
    Nullspace_common_list =[]
    Nullspace_alltask_list = []
    importance_list = []

    for k,ncla in taskcla:
        # specify threshold hyperparameter

        threshold = np.array([0.98] * 5) + task_id*np.array([0.001] * 5)

        task_name = data[k]['name']+'-'+str(k)
        task_name_list.append(task_name)
        print_log('*'*100, log)
        print_log('Task {:2d} ({:s})'.format(k,task_name), log)
        print_log('*'*100, log)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =test_data[k]['test']['x']
        ytest =test_data[k]['test']['y']
        task_list.append(k)

        lr = args.lr 
        best_loss=np.inf
        print_log ('-'*40, log)
        print_log ('Task ID :{} | Learning Rate : {}'.format(task_id, lr), log)
        print_log ('-'*40, log)

        if task_id==0:
            model = LeNet(taskcla).to(device)
            for k_t, (m, param) in enumerate(model.named_parameters()):
                print (k_t,m,param.shape)
            memory[task_name] = {}

            print_log ('Model parameters ---', log)
            kk = 0
            for k_t, (m, param) in enumerate(model.named_parameters()):
                if 'weight' in m and 'bn' not in m:
                    print_log ((k_t,m,param.shape), log)
                    memory[task_name][str(kk)] = {
                        'space_list': {},
                        'grad_list': {},
                        'regime':{},
                    }
                    kk += 1

            print_log ('-'*40, log)
            model.apply(init_weights)
            best_model=get_model(model)
            grad_list =[]
            normal_param = [
                param for name, param in model.named_parameters()
                if not 'scale' in name
            ] 
            optimizer = torch.optim.SGD([{'params': normal_param}],lr=lr)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion, k)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain, criterion, k)
                print_log('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)),log)
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, k)
                print_log(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),log)
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    print_log(' *',log)
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        print_log(' lr={:.1e}'.format(lr), log)
                        if lr<args.lr_min:
                            print_log("", log)
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print_log("", log)
            set_model_(model,best_model)
            # Test
            print_log ('-'*40, log)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            print_log('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc), log)
            # Memory Update  
            grad_list = get_gradient_each_layer (model, device, optimizer, criterion, k, xtrain, ytrain)
            Nullspace_common_list, importance_list, Gradient_alltask_list, Nullspace_alltask_list = get_space_and_grad (model, device, grad_list, threshold, task_name, task_name_list, task_id, Gradient_alltask_list, Nullspace_common_list, importance_list, Nullspace_alltask_list)


        else:
            memory[task_name] = {}
            
            # print("-----------------")
            normal_param = [
                param for name, param in model.named_parameters()
                if not 'scale' in name 
            ] 

            scale_param = [
                param for name, param in model.named_parameters()
                if 'scale' in name 
            ]
            optimizer = torch.optim.SGD([
                                        {'params': normal_param},
                                        {'params': scale_param, 'weight_decay': 0, 'lr':lr}
                                        ],
                                        lr=lr
                                        )

            projection_list = []
            # Projection Matrix Precomputation
            for i in range(len(Nullspace_common_list)):
                projection_operator = torch.mm(Nullspace_common_list[i], torch.mm(torch.diag(importance_list[i].float()), Nullspace_common_list[i].transpose(0,1)))
                projection_list.append(projection_operator)


            #==1 gradient projection condition
            print_log('excute gradient projection condition', log)
            
            

            print_log ('-'*40, log)
            for epoch in range(1, args.n_epochs+1):
                # Train 
                clock0=time.time()
                train_projected_regime(args, model,device,xtrain, ytrain,optimizer,criterion, task_name, task_name_list,grad_list, task_id, projection_list)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,criterion,task_id)
                print_log('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)),log)
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid, criterion, task_id)
                print_log(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),log)
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    print_log(' *',log)
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        print_log(' lr={:.1e}'.format(lr), log)
                        if lr<args.lr_min:
                            print_log("", log)
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print_log("", log)
            set_model_(model,best_model)
            # Test 
            test_acc_sum = 0
            # for i in range(10):
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion,task_id)
            print_log('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc), log)  
            # test_acc_sum = test_acc_sum/10.
            print_log('Average acc={:5.1f}%'.format(test_acc), log)  
            # Memory Update 
            grad_list = get_gradient_each_layer (model, device, optimizer, criterion, k, xtrain, ytrain)
            Nullspace_common_list, importance_list, Gradient_alltask_list, Nullspace_alltask_list = get_space_and_grad (model, device, grad_list, threshold, task_name, task_name_list, task_id, Gradient_alltask_list, Nullspace_common_list, importance_list, Nullspace_alltask_list)
            

        process = psutil.Process()
        rss,vms,_,_,_,_,_ = process.memory_info()
        rss = rss / 1024 / 1024
        vms = vms / 1024 / 1024
        print('Current memory usage: RSS=%.2fMB, VMS=%.2fMB' % (rss, vms))

            
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =test_data[ii]['test']['x']
            ytest =test_data[ii]['test']['y'] 
         

            test_loss, test_acc = test(args, model, device, xtest, ytest,criterion,ii) 
            print_log('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc), log)  
            
            acc_matrix[task_id,jj] = test_acc
            jj +=1
        print_log('Accuracies =', log)
        for i_a in range(task_id+1):
            print_log('\t', log)
            for j_a in range(acc_matrix.shape[1]):
                print_log('{:5.1f}% '.format(acc_matrix[i_a,j_a]), log, end='')
            print_log("", log)
        # update task id 
        task_id +=1
        save_model(model, memory, args.savename)
    print_log('-'*50, log)
    # Simulation Results 
    print_log ('Task Order : {}'.format(np.array(task_list)), log)
    print_log ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()), log) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print_log ('Backward transfer: {:5.2f}%'.format(bwt), log)
    print_log('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000), log)
    print_log('-'*50, log)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index = [i for i in ["1","2","3","4","5","6","7",\
                                        "8","9","10","11","12","13","14","15","16","17","18","19","20"]],
                      columns = [i for i in ["1","2","3","4","5","6","7",\
                                        "8","9","10","11","12","13","14","15","16","17","18","19","20"]])
    sn.set(font_scale=1.4) 
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.show()
def print_log(print_string, log, end=None):
    if end is not None:
        print("{}".format(print_string), end='')
    else:
        print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=50, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--t_order', type=int, default=0, metavar='TOD',
                        help='random seed (default: 0)')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--savename', type=str, default='save/sup',
                        help='save path')

    parser.add_argument('--scale_coff', type=float, default=4.5, metavar='SCF',
                        help='scale co-efficeint (default: 10)')
    parser.add_argument('--gpm_eps', type=float, default=0.98, metavar='EPS',
                        help='threshold (default: 0.97)')
    parser.add_argument('--gpm_eps_inc', type=float, default=0.001, metavar='EPSI',
                        help='threshold increment per task (default: 0.003)')


    args = parser.parse_args()
    if not os.path.exists(args.savename):
       os.makedirs(args.savename)
    log = open(os.path.join(args.savename,
                            'log_seed_{}.txt'.format(args.seed)), 'w')
    print_log('='*100, log)
    print_log('Arguments =', log)
    for arg in vars(args):
        print_log('\t'+arg+': {}'.format(getattr(args,arg)), log)
    print_log('='*100, log)


    print_log('save path : {}'.format(args.savename), log)
    main(args)


