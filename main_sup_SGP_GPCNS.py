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

import random
import pdb
import argparse,time
import math
from copy import deepcopy

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
        self.conv1 = nn.Conv2d(3, 20, 5, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(3)        
        self.map.append(s)
        self.conv2 = nn.Conv2d(20, 50, 5, bias=False, padding=2)
        
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
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
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
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output[task_id], target)         
        loss.backward()
        # Gradient Projections 
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if k<4 and len(params.size())!=1:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                        projection_list[kk]).view(params.size())
                kk +=1
            elif (k<4 and len(params.size())==1) and task_id !=0 :
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
    steps = 1
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125] # Take 125 random samples 
    b = b.cpu()
    example_data = x[b]
    example_data, target = example_data.to(device), y[b].to(device)
    example_out  = net(example_data)
    
    batch_list=[2*12,100,125,125] 
    pad = 2
    p1d = (2, 2, 2, 2)
    mat_list=[]
    act_key=list(net.act.keys())
    grad_list=[] # list contains gradient of each layer

    net.eval()
    example_out  = net(example_data)

    # pdb.set_trace()
    for i in range(len(net.map)):
        bsz=batch_list[i]
        k=0
        if i<2:
            ksz= net.ksize[i]
            s=compute_conv_output_size(net.map[i],net.ksize[i],1,pad)
            mat = np.zeros((net.ksize[i]*net.ksize[i]*net.in_channel[i],s*s*bsz))
            act = F.pad(net.act[act_key[i]], p1d, "constant", 0).detach().cpu().numpy()
         
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) #?
                        k +=1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)


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
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list, grad_list




def update_GPM (args, model, mat_list, grad_list, device, threshold_list, task_id, grad_stack_list, feature_list, space_list, scaling_list, space_hstack_list):
    
    print ('threshold_list: ', threshold_list) 

    if not space_list:
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
            # criteria
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold_list[i]) #+1  
            # update GPM
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
            scaling_list.append(torch.ones(space_list[i].shape[1]).to(device))
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
            # criteria
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold_list[i]) #+1  
            # update GPM
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

            scaling_list[i] = importance
            # print(scaling_list[i])



    print('-'*40)
    print('Common Null Space Matrix')
    print('-'*40)
    for i in range(len(space_list)):
        print ('Layer {} : {} * {}'.format(i+1,space_list[i].shape[0], space_list[i].shape[1]))
    print('-'*40)
    return space_list, feature_list, scaling_list, grad_stack_list, space_hstack_list



def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)
    ## setup seeds
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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
    for k,ncla in taskcla:
        # specify threshold hyperparameter
        threshold = np.array([args.gpm_eps] * 4) + task_id*np.array([args.gpm_eps_inc] * 4)
     
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =test_data[k]['test']['x']
        ytest =test_data[k]['test']['y']
        print(ytrain.shape,yvalid.shape,ytest.shape)

        lr = args.lr 
        best_loss=np.inf
        print ('-'*40)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print ('-'*40)
        
        if task_id==0:
            model = LeNet(taskcla).to(device)
            # print ('Model parameters ---')
            # for k_t, (m, param) in enumerate(model.named_parameters()):
            #     print (k_t,m,param.shape)
            # print ('-'*40)
            # Initialize model 
            model.apply(init_weights)
            best_model=get_model(model)
            space_list =[]
            feature_list = []
            scaling_list = []
            grad_stack_list = []
            space_hstack_list = []
            optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion, k)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)),end='')
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, k)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<args.lr_min:
                            print()
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print()
            set_model_(model,best_model)
            # Test
            print ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory and Importance Update 
            mat_list, grad_list = get_representation_and_gradient (model, device, optimizer, criterion, k, xtrain, ytrain)
            space_list, feature_list, scaling_list, grad_stack_list, space_hstack_list = update_GPM(args, model, mat_list, grad_list, device, threshold, task_id, grad_stack_list, feature_list, space_list, scaling_list, space_hstack_list)

        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            projection_list = []
            # Projection Matrix Precomputation
            for i in range(len(model.act)):
                projection_operator = torch.mm(space_list[i], torch.mm(torch.diag(scaling_list[i].float()), space_list[i].transpose(0,1)))
                # Uf=torch.Tensor(np.dot(feature_list[i],np.dot(np.diag(scaling_list[i]),feature_list[i].transpose()))).to(device) 
                # print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                # Uf.requires_grad = False
                projection_list.append(projection_operator)
                
            print ('-'*40)
            for epoch in range(1, args.n_epochs+1):
                # Train 
                clock0=time.time()
                train_projected(args, model,device,xtrain, ytrain,optimizer,criterion,projection_list,k)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,criterion,k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)),end='')
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid, criterion,k)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<args.lr_min:
                            print()
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print()
            set_model_(model,best_model)
            # Test 
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion,k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))  
            # Memory and Importance Update 
            mat_list, grad_list = get_representation_and_gradient (model, device, optimizer, criterion, k, xtrain, ytrain)
            space_list, feature_list, scaling_list, grad_stack_list, space_hstack_list = update_GPM(args, model, mat_list, grad_list, device, threshold, task_id, grad_stack_list, feature_list, space_list, scaling_list, space_hstack_list)

        # save accuracy 
        jj = 0 
        for ii in task_order[args.t_order][0:task_id+1]:
            xtest =test_data[ii]['test']['x']
            ytest =test_data[ii]['test']['y']
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion,ii) 
            jj +=1
        

        process = psutil.Process()
        rss,vms,_,_,_,_,_ = process.memory_info()
        rss = rss / 1024 / 1024
        vms = vms / 1024 / 1024
        print('Current memory usage: RSS=%.2fMB, VMS=%.2fMB' % (rss, vms))


        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
            print()
        # update task id 
        task_id +=1
    print('-'*50)
    # Simulation Results 
    # print ('Task Order : {}'.format(task_order[args.t_order]))
    # print("Configs: seed: {} | lr: {} | gpm_eps: {} | gpm_eps_inc: {} | scale_coff: {}".format(args.seed,args.lr,args.gpm_eps,args.gpm_eps_inc,args.scale_coff)) 
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='FE-GPCNS (GPM)')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=50, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
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
    # SGP/GPM specific 
    parser.add_argument('--scale_coff', type=int, default=4.5, metavar='SCF',
                        help='scale co-efficeint (default: 10)')
    parser.add_argument('--gpm_eps', type=float, default=0.98, metavar='EPS',
                        help='threshold (default: 0.97)')
    parser.add_argument('--gpm_eps_inc', type=float, default=0.001, metavar='EPSI',
                        help='threshold increment per task (default: 0.003)')

    args = parser.parse_args()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    main(args)



