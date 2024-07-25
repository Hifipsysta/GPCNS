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

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(nn.Module):
    def __init__(self,taskcla):
        super(AlexNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048,2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])
        
        self.taskcla = taskcla
        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048,n,bias=False))
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3']=x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        y=[]
        for t,i in self.taskcla:
            y.append(self.fc3[t](x))
            
        return y

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

        conv_idx = 0
        for k, (m,params) in enumerate(model.named_parameters()):
            
            if conv_idx < 5:
                update = params.grad.data

                if len(update.shape) == 4:
                    update_size =  update.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(update_size,-1), projection_list[conv_idx]).view(params.size())
   
                    conv_idx += 1

                elif len(update.shape)==2:
                    params.grad.data = params.grad.data - torch.mm(params.grad.data, projection_list[conv_idx])
                    conv_idx += 1
            if (k<15 and len(params.size())==1) and task_id !=0 :
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




def get_representation_and_gradient (net, device, x, y=None): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125] # Take 125 random samples 
    b = b.cpu()
    example_data = x[b]
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    batch_list=[2*12,100,100,125,125] 
    mat_list=[]
    act_key=list(net.act.keys())
    for i in range(len(net.map)):
        bsz=batch_list[i]
        k=0
        if i<3:
            ksz= net.ksize[i]
            s=compute_conv_output_size(net.map[i],net.ksize[i])
            mat = np.zeros((net.ksize[i]*net.ksize[i]*net.in_channel[i],s*s*bsz))
            act = net.act[act_key[i]].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) 
                        k +=1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)


    conv_idx = 0
    grad_list = []
    for k, (m,params) in enumerate(net.named_parameters()):
        if conv_idx < 5:
            update = params.grad.data 
            update_size = update.shape[0]
            if len(update.shape) == 4:
                grad_list.append(params.grad.data.view(update_size,-1))
                conv_idx += 1

            elif len(update.shape)==2:
                grad_list.append(update)
                conv_idx += 1
        else:
            break  


    print('#####len(mat_list)#####len(grad_list)', len(mat_list), len(grad_list))

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list, grad_list


def update_bases (args, model, mat_list, grad_list, device, threshold_list, task_id, grad_stack_list, feature_list, space_list, importance_list, space_hstack_list):
    
    print ('threshold_list: ', threshold_list) 

    if not space_list:
        # After First Task 
        for i in range(len(grad_list)):
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
    print (device)
    ## setup seeds
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ## Load CIFAR100 DATASET
    from dataloader import cifar100 as cf100
    data,taskcla,inputsize=cf100.get(pc_valid=args.pc_valid)

    acc_matrix=np.zeros((10,10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    for k,ncla in taskcla:
        # specify threshold hyperparameter
        threshold = np.array([args.gpm_eps] * 5) + task_id * np.array([args.gpm_eps_inc] * 5)
        # threshold = [0.997, 0.997, 0.997, 0.997, 0.997]
     
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        lr = args.lr 
        best_loss=np.inf
        print ('-'*40)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print ('-'*40)
        
        if task_id==0:
            model = AlexNet(taskcla).to(device)
            # print ('Model parameters ---')
            # for k_t, (m, param) in enumerate(model.named_parameters()):
            #     print (k_t,m,param.shape)
            # print ('-'*40)

            best_model=get_model(model)
            space_list =[]
            feature_list = []
            importance_list = []
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
            mat_list, grad_list = get_representation_and_gradient(model, device, xtrain, ytrain)
            space_list, feature_list, importance_list, grad_stack_list, space_hstack_list = update_bases(args, model, mat_list, grad_list, device, threshold, task_id, grad_stack_list, feature_list, space_list, importance_list, space_hstack_list)

        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            projection_list = []
            # Projection Matrix Precomputation
            for i in range(len(model.act)):
                projection_operator = torch.mm(space_list[i], torch.mm(torch.diag(importance_list[i].float()), space_list[i].transpose(0,1)))
                # Uf=torch.Tensor(np.dot(feature_list[i],np.dot(np.diag(importance_list[i]),feature_list[i].transpose()))).to(device) 
                # print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                # Uf.requires_grad = False
                projection_list.append(projection_operator)
                
            # print ('-'*40)
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
            mat_list, grad_list = get_representation_and_gradient(model, device, xtrain, ytrain)
            space_list, feature_list, importance_list, grad_stack_list, space_hstack_list = update_bases(args, model, mat_list, grad_list, device, threshold, task_id, grad_stack_list, feature_list, space_list, importance_list, space_hstack_list)


        
        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
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
    # print ('Task Order : {}'.format(np.array(task_list)))
    # print("Configs: seed: {} | lr: {} | gpm_eps: {} | gpm_eps_inc: {} | scale_coff: {}".format(args.seed,args.lr,args.gpm_eps,args.gpm_eps_inc,args.scale_coff)) 
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)



if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='10-split CIFAR-100 with FE-GPCNS (GPM)')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=5, metavar='S',
                        help='random seed (default: 5)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=5e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')

    parser.add_argument('--scale_coff', type=float, default=1.5, metavar='SCF',
                        help='importance co-efficeint (default: 10)')
    parser.add_argument('--gpm_eps', type=float, default=0.97, metavar='EPS',
                        help='threshold (default: 0.97)')
    parser.add_argument('--gpm_eps_inc', type=float, default=0.003, metavar='EPSI',
                        help='threshold increment per task (default: 0.003)')


    args = parser.parse_args()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    main(args)



