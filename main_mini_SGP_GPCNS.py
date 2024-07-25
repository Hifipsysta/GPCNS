import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
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
# from layers import Conv2d, Linear



## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x
        self.count +=1
        out = relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)            
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 2)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 9, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        
        bsz = x.size(0)
        
        # print(x.shape)
        self.act['conv_in'] = x.view(bsz, 3, 84, 84)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 84, 84)))) 
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = avg_pool2d(out, 2)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        y=[]
        for t,i in self.taskcla:
            y.append(self.linear[t](out))      
        return y

def ResNet18(taskcla, nf=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla, nf)

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def save_model(model, savename):
    ckpt = {
        'model': model.state_dict(),
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

def train_projected(args,model,device,x,y,optimizer,criterion,projection_list,task_id):
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
        # print(i)
        
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output[task_id], target)         
        loss.backward()

        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if len(params.size())==4 and 'weight' in m:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                    projection_list[kk]).view(params.size())

                kk+=1
            elif len(params.size())==1 and task_id !=0:
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

def get_representation_and_gradient (net, device, optimizer, criterion, task_id, x, y=None): 
    # Collect activations by forward pass
    
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:100] # ns=100 examples 
    b = b.cpu()
    example_data = x[b]
    example_data, target = example_data.to(device), y[b].to(device)
    grad_list=[] # list contains gradient of each layer


    net.eval()
    example_out  = net(example_data)
    
    act_list =[]
    act_list.extend([net.act['conv_in'], 
        net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
        net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
        net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
        net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])

    batch_list  = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100] #scaled
    # network arch 

    stride_list = [2, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
    map_list    = [84, 42,42,42,42, 42,21,21,21, 21,11,11,11, 11,6,6,6] 
    in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160] 
    pad = 1
    sc_list=[5,9,13]
    p1d = (1, 1, 1, 1)
    mat_final=[] # list containing GPM Matrices 
    mat_list=[]
    mat_sc_list=[]
    for i in range(len(stride_list)):
        if i==0:
            ksz = 3
        else:
            ksz = 3 
        bsz=batch_list[i]
        st = stride_list[i]     
        k=0
        s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
        mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                    k +=1
        mat_list.append(mat)
        # For Shortcut Connection
        if i in sc_list:
            k=0
            s=compute_conv_output_size(map_list[i],1,stride_list[i])
            mat = np.zeros((1*1*in_channel[i],s*s*bsz))
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                        k +=1
            mat_sc_list.append(mat) 

    ik=0
    for i in range (len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6,10,14]:
            mat_final.append(mat_sc_list[ik])
            ik+=1

    k_conv = 0
    for k, (m,params) in enumerate(net.named_parameters()):
        if len(params.shape) == 4 and 'weight' in m:
            
            grad = params.grad.data
            grad = grad.reshape(grad.shape[0], grad.shape[1]*grad.shape[2]*grad.shape[3])
            grad_list.append(grad)
            k_conv += 1

    print('-'*30)
    print('Representation Matrix and Gradient Matrix')
    print('-'*30)
    for i in range(len(mat_final)):
        print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
        print ('Layer {} : {}'.format(i+1,grad_list[i].shape))

    print('-'*30)

    return mat_final, grad_list



def update_bases (args, model, mat_list, grad_list, device, threshold_list, task_id, grad_stack_list, feature_list, space_list, importance_list, space_hstack_list):
    print ('Threshold: ', threshold_list) 

    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            Gradient_matrix = grad_list[i]
            print('Gradient Matrix', Gradient_matrix.shape)
            grad_stack_list.append(Gradient_matrix)

            U_matrix, Sigma, Vh_matrix = torch.linalg.svd(grad_stack_list[i], full_matrices=False)
            V_matrix = Vh_matrix.transpose(0,1)
            
            Sigma_total = torch.sum(Sigma)
            Sigma_accumul_list = [torch.sum(Sigma[0:idx+1]) for idx in range(len(Sigma))]
            Sigma_select = torch.tensor(Sigma_accumul_list).to(device) <= threshold_list[i]*Sigma_total
            Sigma_slim = Sigma[Sigma_select]
            rank_subspace = torch.count_nonzero(Sigma_slim)
            print('Rank of Matrix：', rank_subspace)

            space_list.append(V_matrix[:, :rank_subspace])
            print('space_list[i]', space_list[i].shape)


            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold_list[i]) 
            feature_list.append(torch.tensor(U[:,0:r]).float().to(device))


            space_hstack_list.append(torch.hstack([space_list[i], feature_list[i]]))
            U_matrix, Sigma, Vh_matrix = torch.linalg.svd(space_hstack_list[i], full_matrices=False)

            Sigma_total = torch.sum(Sigma)
            Sigma_accumul_list = [torch.sum(Sigma[0:idx+1]).item() for idx in range(len(Sigma))]
            Sigma_select = torch.tensor(Sigma_accumul_list).to(device) <= threshold_list[i]*Sigma_total
            Sigma_slim = Sigma[Sigma_select]
            rank_subspace = torch.count_nonzero(Sigma_slim)


            space_list[i] = U_matrix[:, :rank_subspace]
            print('Revised space_list[i]', space_list[i].shape)


            importance_list.append(torch.ones(space_list[i].shape[1]).to(device))
    else:
        for i in range(len(grad_list)):
            Gradient_matrix = grad_list[i]
            print('Gradient Matrix', Gradient_matrix.shape)
            grad_stack_list[i] = torch.vstack([grad_stack_list[i], Gradient_matrix])   
            print('Matrix waiting to be decomposed', grad_stack_list[i].shape)

            U_matrix, Sigma, Vh_matrix = torch.linalg.svd(grad_stack_list[i], full_matrices=False) 
            V_matrix = Vh_matrix.transpose(0,1)

            Sigma_total = torch.sum(Sigma)
            Sigma_accumul_list = [torch.sum(Sigma[0:idx+1]).item() for idx in range(len(Sigma))]
            Sigma_select = torch.tensor(Sigma_accumul_list).to(device) <= threshold_list[i]*Sigma_total
            Sigma_slim = Sigma[Sigma_select]
            rank_subspace = torch.count_nonzero(Sigma_slim)
            print('Rank of Matrix：', rank_subspace)

            space_list[i] = V_matrix[:, :rank_subspace]
            print('space_list[i]', space_list[i].shape)


            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold_list[i])   
            feature_list[i] = torch.hstack([feature_list[i], torch.tensor(U[:,0:r]).float().to(device)])




            space_hstack_list[i] = torch.hstack([space_list[i], feature_list[i]])
            U_matrix, Sigma, Vh_matrix = torch.linalg.svd(space_hstack_list[i], full_matrices=False)


            Sigma_total = torch.sum(Sigma)
            Sigma_accumul_list = [torch.sum(Sigma[0:idx+1]).item() for idx in range(len(Sigma))]
            Sigma_select = torch.tensor(Sigma_accumul_list).to(device) <= threshold_list[i]*Sigma_total
            Sigma_slim = Sigma[Sigma_select]
            rank_subspace = torch.count_nonzero(Sigma_slim)

            space_list[i] = U_matrix[:, :rank_subspace]
            print('Revised space_list[i]', space_list[i].shape)
            

            importance = ((args.scale_coff+1)*Sigma[:rank_subspace])/(args.scale_coff*Sigma[:rank_subspace] + max(Sigma[:rank_subspace]))         

            importance_list[i] = importance
            # print(importance_list[i])

    print('-'*40)
    print('Common Null Space Matrix')
    print('-'*40)
    for i in range(len(space_list)):
        print ('Layer {} : {} * {}'.format(i+1,space_list[i].shape[0], space_list[i].shape[1]))
    print('-'*40)
    return space_list, feature_list, importance_list, grad_stack_list, space_hstack_list 





def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ## Load CIFAR100 DATASET
    from dataloader import miniimagenet as data_loader
    dataloader = data_loader.DatasetGen(args)
    taskcla, inputsize = dataloader.taskcla, dataloader.inputsize

    acc_matrix=np.zeros((20,20))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    task_name_list = []
    acc_list_all = []

    importance_list = []


    for k,ncla in taskcla:
        # specify threshold hyperparameter

        threshold = np.array([0.985] * 20)
        data = dataloader.get(k)
        task_name = data[k]['name']
        task_name_list.append(task_name)
        print_log('*'*100, log)
        print_log('Task {:2d} ({:s})'.format(k,data[k]['name']), log)
        print_log('*'*100, log)
        

        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        lr = args.lr 
        best_loss=np.inf
        print_log ('-'*40, log)
        print_log ('Task ID :{} | Learning Rate : {}'.format(task_id, lr), log)
        print_log ('-'*40, log)
        
        if task_id==0:
            model = ResNet18(taskcla,20).to(device) # base filters: 20

            print_log ('-'*40, log)

            best_model=get_model(model)
            space_list =[]
            feature_list = []
            importance_list = []
            grad_stack_list = []
            space_hstack_list = []

            # normal_param = [
            #     param for name, param in model.named_parameters()
            #     if not 'scale' in name
            # ] 
            # optimizer = torch.optim.SGD([{'params': normal_param}],lr=lr)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            acc_list = []
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
                acc_list.append(valid_acc)
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
            print(acc_list)
            acc_list_all.append(acc_list)
            set_model_(model,best_model)
            # Test
            print_log ('-'*40, log)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            print_log('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc), log)
            # Memory Update  
            mat_list,grad_list = get_representation_and_gradient (model, device, optimizer, criterion, k, xtrain, ytrain)
            space_list, feature_list, importance_list, grad_stack_list, space_hstack_list = update_bases(args, model, mat_list, grad_list, device, threshold, task_id, grad_stack_list, feature_list, space_list, importance_list, space_hstack_list)
            


        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            projection_list = []
            # Projection Matrix Precomputation
            for i in range(len(space_list)):
                projection_operator = torch.mm(space_list[i], torch.mm(torch.diag(importance_list[i].float()), space_list[i].transpose(0,1)))
                # Uf=torch.Tensor(np.dot(feature_list[i],np.dot(np.diag(importance_list[i]),feature_list[i].transpose()))).to(device) 
                # print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                # Uf.requires_grad = False
                projection_list.append(projection_operator)

            print ('-'*40)


            acc_list = []
            

            for epoch in range(1, args.n_epochs+1):
                # Train 

                clock0=time.time()
                train_projected(args, model,device,xtrain, ytrain,optimizer,criterion,projection_list,k)

                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,criterion,task_id)
                print_log('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)),log)
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid, criterion, task_id)
                acc_list.append(valid_acc)
     
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
            print(acc_list)
            acc_list_all.append(acc_list)
            set_model_(model,best_model)
            # Test 
            test_acc_sum = 0
            for i in range(10):
                test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion,task_id)
                print_log('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc), log)  
                test_acc_sum += test_acc
            test_acc_sum = test_acc_sum/10.
            print_log('Average acc={:5.1f}%'.format(test_acc_sum), log)  
            # Memory Update 
            mat_list,grad_list = get_representation_and_gradient (model, device, optimizer, criterion, task_id, xtrain, ytrain)
            space_list, feature_list, importance_list, grad_stack_list, space_hstack_list = update_bases(args, model, mat_list, grad_list, device, threshold, task_id, grad_stack_list, feature_list, space_list, importance_list, space_hstack_list)

            


        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            
            task_test = data[ii]['name']
            print_log('current testing task:{}'.format(task_test), log)

             
         
            test_acc_sum = 0
            for i in range(5):
                test_loss, test_acc = test(args, model, device, xtest, ytest,criterion,ii) 
                print_log('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc), log)  
                test_acc_sum += test_acc
            acc_matrix[task_id,jj] = test_acc_sum/5.
            jj +=1
        

        process = psutil.Process()
        rss,vms,_,_,_,_,_ = process.memory_info()
        rss = rss / 1024 / 1024
        vms = vms / 1024 / 1024
        print('Current memory usage: RSS=%.2fMB, VMS=%.2fMB' % (rss, vms))


        print_log('Accuracies =', log)
        for i_a in range(task_id+1):
            print_log('\t', log)
            for j_a in range(acc_matrix.shape[1]):
                print_log('{:5.1f}% '.format(acc_matrix[i_a,j_a]), log, end='')
            print_log("", log)
        # update task id 
        task_id +=1
        save_model(model,args.savename)
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
    parser = argparse.ArgumentParser(description='5 datasets with FE-GPCNS (GPM)')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=37, metavar='S',
                        help='random seed (default: 37)')
    parser.add_argument('--pc_valid',default=0.02,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-3, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=5, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=3, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--savename', type=str, default='save/five/Ours/test_task2',
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



