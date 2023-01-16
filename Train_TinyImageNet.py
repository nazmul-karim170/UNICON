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
from dataloader_tiny import tinyImagenet_dataloader as dataloader
import copy 
from PreResNet_tiny import *
from Contrastive_loss import *

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.005, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=50, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--id', default='TinyImage')
parser.add_argument('--data_path', default='./data/tiny-imagenet-200', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--tau',  default=5, type=float)
parser.add_argument('--ratio', default=0.2 , type=float, help='noise ratio')
parser.add_argument('--resume', default=False , type=bool, help='Resume from chekpoint')
parser.add_argument('--num_class', default=200, type=int)
parser.add_argument('--dataset', default='TinyImageNet', type=str)

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    # Freeze one network and Train the other
    net2.eval()         
    net.train()

    ## loss metrics
    loss_x = 0
    loss_u = 0
    loss_ucl = 0

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        
        batch_size = inputs_x.size(0)

        # Transform label to One-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
        
        with torch.no_grad():

            # Pseudo-label
            _, outputs_u11 = net(inputs_u)
            _, outputs_u12 = net(inputs_u2)
            _, outputs_u21 = net2(inputs_u)
            _, outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       

            ptu = pu**(1/args.T)            ## Temparature Sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # Normalize
            targets_u = targets_u.detach()                  

            ## Label Refinement
            _, outputs_x  = net(inputs_x)
            _, outputs_x2 = net(inputs_x2)           

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T)            ## Temparature sharpening 
                        
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)           
            targets_x = targets_x.detach()
 

        ## Mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)

        ## Unsupervised Contrastive Loss
        f1, _ = net(inputs_u3)
        f2, _ = net(inputs_u4)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss_simCLR = contrastive_criterion(features)

        all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b   = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        ## Mixing inputs 
        mixed_input  = l * input_a  + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        _, logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
        
        ## Semi-supervised Loss
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # Regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        ## Total Loss
        loss = Lx + lamb * Lu + args.lambda_c * loss_simCLR + penalty  

        loss_x += Lx.item()
        loss_u += Lu.item()
        loss_ucl += loss_simCLR.item()

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f Contrastive Loss:%.4f'
                %(args.dataset, args.ratio, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss_x/(batch_idx+1), loss_u/(batch_idx+1), loss_ucl/(batch_idx+1)))
        sys.stdout.flush()


## Warm-Up Model
def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1

    for batch_idx, (inputs, labels, index) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _, outputs = net(inputs)               
        loss = CEloss(outputs, labels)     

        if args.noise_mode=='asym':  # Penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        else:
            L = loss

        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.ratio, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

## Validation    
def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _,outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
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
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1 = net1(inputs)       
            _, outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            loss = CEloss(outputs, targets)  
            loss_x += loss.item()
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc)) 
    return acc    

# Calculate the KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## JSD Calculation
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)
 
## JSD Calculation
def Calculate_JSD(model1, model2, num_samples):  
    JS_dist = Jensen_Shannon()
    JSD   = torch.zeros(num_samples)    

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
            out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])

        ## Get the Prediction
        out = (out1 + out2)/2     

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(out,  F.one_hot(targets, num_classes = args.num_class))
        JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist

    return JSD

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
               
def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

folder = 'TinyImageNet_' + str(args.ratio)
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

log=open(os.path.join(model_save_loc, 'test_acc_%s.txt'%args.id),'w')     
log.flush()

warm_up = 15
loader = dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=4, log = log, ratio = args.ratio, noise_mode = args.noise_mode, noise_file='%s/clean_%.2f_%s.npz'%(args.data_path,args.ratio, args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True


## Loss Functions and Optimizers 
criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
contrastive_criterion = SupConLoss()


        ## Resume From Warmup Checkpoint ##
model_name_1 = 'Net1_warmup.pth'
model_name_2 = 'Net2_warmup.pth'

if args.resume:
    start_epoch = warm_up
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1))['net'])
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2))['net'])
else:
    start_epoch = 0

best_acc = 0

## Dummy Sample Ratio
SR = 0
lr = args.lr

## Main Training 
for epoch in range(start_epoch,args.num_epochs+1):   
    num_samples = 100000

    ## After 100 epochs, change the learning rate of the optimizer  
    if (epoch+1)%200 == 0:
        lr /= 10

    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr   

    test_loader = loader.run(SR, 'val')
    acc = test(net1,net2, test_loader)
    log.write(str(acc)+'\n')
    log.flush()  

    ## Warmup Stage 
    if epoch<warm_up:       
        warmup_trainloader = loader.run(SR, 'warmup')

        print('Warmup Net 1')
        warmup(epoch, net1, optimizer1, warmup_trainloader)   

        print('\nWarmup Net 2')
        warmup(epoch, net2, optimizer2, warmup_trainloader) 
    
    else:
        eval_loader = loader.run(SR,'eval_train')  

        prob = Calculate_JSD(net1, net2, num_samples)
        threshold = torch.mean(prob)
        print("Threshold:", threshold)
        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-torch.min(prob))/arg.tau
        SR = torch.sum(prob<threshold).item()/num_samples
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob)        # Uniform Selection
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)            # Train net1  

        prob = Calculate_JSD(net2, net1, num_samples)           
        threshold = torch.mean(prob)
        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-torch.min(prob))/arg.tau
        SR = torch.sum(prob<threshold).item()/num_samples            

        print('\n Train Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob= prob)    # Uniform Selection
        train(epoch, net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader)       # train net1

    acc = test(net1,net2, test_loader)
    log.write(str(acc)+'\n')
    log.flush()  

    if acc > best_acc:
        if epoch <warm_up:
            model_name_1 = 'Net1_warmup.pth'
            model_name_2 = 'Net2_warmup.pth'
        else:
            model_name_1 = 'Net1.pth'
            model_name_2 = 'Net2.pth'            

        print("Save the Model-----")
        checkpoint1 = {
            'net': net1.state_dict(),
            'Model_number': 1,
            'Noise_Ratio': args.ratio,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Dataset': 'TinyImageNet',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        checkpoint2 = {
            'net': net2.state_dict(),
            'Model_number': 2,
            'Noise_Ratio': args.ratio,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Dataset': 'TinyImageNet',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))
        best_acc = acc
