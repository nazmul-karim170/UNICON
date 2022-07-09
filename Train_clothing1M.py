from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader

from sklearn.mixture import GaussianMixture
import copy 
import torchnet
from Contrastive_loss import *
from PreResNet_source import *


parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')    
parser.add_argument('--lr', '--learning_rate', default=0.005, type=float, help='initial learning rate')   ## Set the learning rate to 0.005 for faster training at the beginning
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--data_path', default='./data/Clothing1M_org', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=1000, type=int)
parser.add_argument('--dataset', default="Clothing1M", type=str)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
contrastive_criterion = SupConLoss()

## For plotting the logs
# import wandb
# wandb.init(project="noisy-label-project-clothing1M", entity="...")

## Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net2.eval()         # Fix one network and train the other    
    net.train()       

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter_un  = (len(unlabeled_trainloader.dataset)//args.batch_size)+1
    num_iter_lab = (len(labeled_trainloader.dataset)//args.batch_size)+1

    loss_x = 0
    loss_u = 0
    loss_scl = 0
    loss_ucl = 0

    num_iter = num_iter_lab
    
    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
        
        with torch.no_grad():

            # Label co-guessing of unlabeled samples
            _, outputs_u11 = net(inputs_u3)
            _, outputs_u12 = net(inputs_u4)
            _, outputs_u21 = net2(inputs_u3)
            _, outputs_u22 = net2(inputs_u4)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T)                             ## Temparature Sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)   ## Normalize
            targets_u = targets_u.detach()     

            ## Label refinement of labeled samples
            _, outputs_x  = net(inputs_x3)
            _, outputs_x2 = net(inputs_x4)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T)                            ## Temparature sharpening 
                        
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  ## normalize           
            targets_x = targets_x.detach()

        ## Mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l,1-l)

        ## Unsupervised Contrastive Loss
        f1, _ = net(inputs_u)
        f2, _ = net(inputs_u2)
        f1    = F.normalize(f1, dim=1)
        f2    = F.normalize(f2, dim=1)
        features    = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss_simCLR = contrastive_criterion(features)


        all_inputs  = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a , input_b   = all_inputs , all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        ## Mixing inputs 
        mixed_input  = (l * input_a[: batch_size * 2] + (1 - l) * input_b[: batch_size * 2])
        mixed_target = (l * target_a[: batch_size * 2] + (1 - l) * target_b[: batch_size * 2])
                
        _, logits = net(mixed_input)

        Lx = -torch.mean(
            torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1)
        )

        ## Regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx  + args.lambda_c*loss_simCLR + penalty 
        loss_x += Lx.item()
        loss_ucl += loss_simCLR.item()

        ## Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Contrative Loss:%.4f'
                %(args.dataset,  epoch, args.num_epochs, batch_idx+1, num_iter, loss_x/(batch_idx+1), loss_ucl/(batch_idx+1)))
        sys.stdout.flush()


def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _ , outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))
        sys.stdout.flush()
    
def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _ , outputs   = net(inputs)
            _ , predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Acc: %.2f%%" %(k,acc))  
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...'%k)
        save_point = os.path.join(model_save_loc, '%s_net%d.pth.tar'%(args.id,k))
        torch.save(net.state_dict(), save_point)
    return acc

def test(net1,net2,test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1  = net1(inputs)       
            _, outputs2  = net2(inputs)           
            outputs      = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total   += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()        
            acc_meter.add(outputs,targets)
            
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    accs = acc_meter.value()
    return acc , accs   

## Calculate the KL Divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen_Shannon divergence (Symmetric and Smoother than the KL divergence) 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

## Calculate JSD
def Calculate_JSD(epoch,model1, model2):
    model1.eval()
    model2.eval()
    num_samples = args.num_batches*args.batch_size
    prob = torch.zeros(num_samples)
    JS_dist = Jensen_Shannon()
    paths = []
    n=0
    for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda() 
        batch_size      = inputs.size()[0]

        ## Get the output of the Model
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
            out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])

        ## Get the Prediction
        out = 0.5*out1 + 0.5*out2          

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(out,  F.one_hot(targets, num_classes = args.num_class))
        prob[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist 
        
        for b in range(inputs.size(0)):
            paths.append(path[b])
            n+=1

        sys.stdout.write('\r')
        sys.stdout.write('| Evaluating loss Iter %3d\t' %(batch_idx)) 
        sys.stdout.flush()
            
    return prob,paths  

## Penalty for Asymmetric Noise    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

## Get the pre-trained model                
def get_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.num_class)
    return model 

def create_model():
    model = resnet50(num_classes=args.num_class)
    model = model.cuda()
    return model

## Threshold Adjustment 
def linear_rampup(current, warm_up, rampup_length=5):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

## Semi-Supervised Loss
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)


log = open('./checkpoint/%s.txt'%args.id,'w')     
log.flush()

loader = dataloader.clothing_dataloader(root=args.data_path, batch_size=args.batch_size, warmup_batch_size = args.batch_size*2, num_workers=8, num_batches=args.num_batches)
print('| Building Net')

model = get_model()
net1  = create_model()
net2  = create_model()
cudnn.benchmark = True

## Optimizer and Learning Rate Scheduler 
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 100, 1e-5)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 100, 1e-5)

## Cross-Entropy and Other Losses
CE     = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
criterion    = SemiLoss()

## Warm-up Epochs (maximum value is 2, we recommend 0 or 1)
warm_up = 0

## Copy Saved Data
if args.pretrained: 
    params  = model.named_parameters()
    params1 = net1.named_parameters() 
    params2 = net2.named_parameters()

    dict_params2 = dict(params2)
    dict_params1 = dict(params1)

    for name1, param in params:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param.data)
            dict_params1[name1].data.copy_(param.data)


## Location for saving the models 
folder = 'Clothing1M'
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

net1 = nn.DataParallel(net1)
net2 = nn.DataParallel(net2)

## Loading Saved Weights
model_name_1 = 'clothing1m_net1.pth.tar'
model_name_2 = 'clothing1m_net2.pth.tar'

if args.resume:
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1)))
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2)))

best_acc = [0,0]
SR = 0
torch.backends.cudnn.benchmark = True
acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
nb_repeat = 2

for epoch in range(0, args.num_epochs+1):   
    val_loader = loader.run(0, 'val')
    
    if epoch>100:
        nb_repeat =3  ## Change how many times we want to repeat on the same selection

    if epoch<warm_up:             
        train_loader = loader.run(0,'warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,train_loader)
        
        print('\nWarmup Net2')
        train_loader = loader.run(0,'warmup')
        warmup(net2, optimizer2, train_loader)

    else:
        num_samples = args.num_batches*args.batch_size
        eval_loader = loader.run(0.5,'eval_train')  
        prob2, paths2 = Calculate_JSD(epoch, net1, net2)                          ## Calculate the JSD distances 
        threshold   = torch.mean(prob2)                                           ## Simply Take the average as the threshold
        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-torch.min(JSD))/args.tau

        SR = torch.sum(prob2<threshold).item()/prob2.size()[0]                    ## Calculate the Ratio of clean samples      
        
        for i in range(nb_repeat):
            print('\n\nTrain Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob=prob2,  paths=paths2)         ## Uniform Selection
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)                        ## Train Net1
            acc1 = val(net1,val_loader,1)

        print('\n==== Net 1 evaluate next epoch training data loss ====') 
        eval_loader   = loader.run(SR,'eval_train')
        net1.load_state_dict(torch.load(os.path.join(model_save_loc, '%s_net1.pth.tar'%args.id)))
        prob1, paths1 = Calculate_JSD(epoch,net2, net1)  
        threshold     = torch.mean(prob1)
        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-torch.min(JSD))/args.tau   
        SR = torch.sum(prob1<threshold).item()/prob1.size()[0]

        for i in range(nb_repeat):
            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob=prob1, paths=paths1)           ## Uniform Selection
            train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)                             ## Train net2
            acc2 = val(net2,val_loader,2)

    scheduler1.step()
    scheduler2.step()        
    acc1 = val(net1,val_loader,1)
    acc2 = val(net2,val_loader,2)   
    log.write('Validation Epoch:%d  Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))

    net1.load_state_dict(torch.load(os.path.join(model_save_loc, '%s_net1.pth.tar'%args.id)))
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, '%s_net2.pth.tar'%args.id)))
    log.flush() 
    test_loader = loader.run(0,'test')  
    acc, accs = test(net1,net2,test_loader)   
    print('\n| Epoch:%d \t  Acc: %.2f%% (%.2f%%) \n'%(epoch,accs[0],accs[1]))
    log.write('Epoch:%d \t  Acc: %.2f%% (%.2f%%) \n'%(epoch,accs[0],accs[1]))
    log.flush()  

    if epoch<warm_up: 
        model_name_1 = 'Net1_warmup_pretrained.pth'     
        model_name_2 = 'Net2_warmup_pretrained.pth' 

        print("Save the Warmup Model --- --")
        checkpoint1 = {
            'net': net1.state_dict(),
            'Model_number': 1,
            'epoch': epoch,
        }
        checkpoint2 = {
            'net': net2.state_dict(),
            'Model_number': 2,
            'epoch': epoch,
        }
        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))   
