from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
from autoaugment import CIFAR10Policy, ImageNetPolicy

transform_weak_c1m_c10_compose = transforms.Compose(
    [
        transforms.Resize(320),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ]
)


def transform_weak_c1m(x):
    return transform_weak_c1m_c10_compose(x)


transform_strong_c1m_c10_compose = transforms.Compose(
    [
        transforms.Resize(320),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ]
)


def transform_strong_c1m_c10(x):
    return transform_strong_c1m_c10_compose(x)


transform_strong_c1m_in_compose = transforms.Compose(
    [
        transforms.Resize(320),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ]
)
def transform_strong_c1m_in(x):
    return transform_strong_c1m_in_compose(x)

class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = os.path.join(root_dir, 'Imagenet_val')
        self.transform = transform
        self.val_data = []
        with open(os.path.join(root_dir, 'info/synsets.txt')) as f:
            lines = f.readlines()
        synsets = [x.split()[0] for x in lines]
        for c in range(num_class):
            class_path = os.path.join(self.root, synsets[c])
            imgs = os.listdir(class_path)
            for img in imgs:
                self.val_data.append([c, os.path.join(class_path, img)])              
                
    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')   
        img = self.transform(image) 
        return img, target
    
    def __len__(self):
        return len(self.val_data)

class webvision_dataset(Dataset): 
    def __init__(self, root_dir, transform, sample_ratio, mode, num_class, pred=[], probability=[], log=''): 
        self.root = root_dir
        self.transform = transform
        self.mode = mode  
        self.sample_ratio = sample_ratio
        save_file = 'pred_idx_WebVision.npz'
     
        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target                             
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.train_labels = {}
            num_samples  = 0
            self.class_ind = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    train_imgs.append(img)
                    self.train_labels[img] = target
                    num_samples += 1

            ## Get the class indices
            for kk in range(num_class):
                self.class_ind[kk] = [i for i,x in enumerate(train_imgs) if self.train_labels[x]==kk]

            # print(self.class_ind)
            if self.mode == 'all':
                self.train_imgs = train_imgs
            else:                   
                if self.mode == "labeled":
                    pred_idx  = np.zeros(int(self.sample_ratio*num_samples))
                    class_len = int(self.sample_ratio*num_samples/num_class)
                    size_pred  = 0

                    # ## Creating the Class Balance
                    # for i in range(num_class):
                    #         ## Ranking-based selection
                    #     prob1  = np.argsort(probability[self.class_ind[i]])
                    #     print("Indices Size: ", np.shape(prob1))
                    #     class_indices = np.array(self.class_ind[i])
                    #     pred_idx[i*class_len:(i+1)*class_len] = class_indices[prob1[0:class_len]].squeeze()

                    ## Creating the Class Balance
                    for i in range(num_class):
                        sorted_indices  = np.argsort(probability[self.class_ind[i]])      ##  Sorted indices for each class  
                        class_indices   = np.array(self.class_ind[i])                     ##  Class indices  
                        # print(class_indices)
                        size1 = len(class_indices)
                        try:
                            pred_idx[size_pred:size_pred+class_len] = class_indices[sorted_indices[0:class_len].cpu().numpy()].squeeze()
                            size_pred += class_len
                        except:
                            pred_idx[size_pred:size_pred+size1] = np.array(class_indices)
                            size_pred += size1

                    pred_idx = [int(x) for x in list(pred_idx)]
                    np.savez(save_file, index = pred_idx)
                    self.train_imgs = [train_imgs[i] for i in pred_idx]           
                    probability[probability<0.5] = 0                        ## Weight Adjustment 
                    self.probability = [1-probability[i] for i in pred_idx]
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))            

                elif self.mode == "unlabeled":
                    pred_idx1 = np.load(save_file)['index']
                    idx = list(range(num_samples))
                    pred_idx = [x for x in idx if x not in pred_idx1] 
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                         
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))        
                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](image)
                img4 = self.transform[3](image)
            return img1, img2, img3, img4, target, prob

        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](image)
                img4 = self.transform[3](image)
            return img1, img2, img3, img4

        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, index     

        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    


class webvision_dataloader():  
    def __init__(self, batch_size, num_class, num_workers, root_dir, log):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log

        # self.transform_train = transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        #     ]) 
        # self.transform_test = transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        #     ])  
        
        self.transform_imagenet = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])

        self.transforms = {
            "warmup": transform_weak_c1m,
            "unlabeled": [
                        transform_strong_c1m_in,
                        transform_strong_c1m_in,
                        transform_weak_c1m,
                        transform_weak_c1m
                    ],
            "labeled": [
                        transform_strong_c1m_in,
                        transform_strong_c1m_in,
                        transform_weak_c1m,
                        transform_weak_c1m
                    ],
            "test": None,
        }
        self.transforms["test"] = transforms.Compose(
            [
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]
        )
    def run(self,SR, mode, pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = webvision_dataset(root_dir=self.root_dir, transform = self.transforms["warmup"], sample_ratio = SR, mode="all", num_class=self.num_class)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*4,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)                 
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = webvision_dataset(root_dir=self.root_dir, transform = self.transforms["labeled"],  sample_ratio = SR, mode="labeled",num_class=self.num_class,pred=pred,probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)        
            
            unlabeled_dataset = webvision_dataset(root_dir=self.root_dir, transform = self.transforms["unlabeled"],  sample_ratio = SR, mode="unlabeled",num_class=self.num_class,pred=pred,log=self.log)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = webvision_dataset(root_dir=self.root_dir, transform = self.transforms["test"],  sample_ratio = SR, mode='test', num_class=self.num_class)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = webvision_dataset(root_dir=self.root_dir, transform = self.transforms["test"],  sample_ratio = SR, mode='all', num_class=self.num_class)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return eval_loader     
        
        elif mode=='imagenet':
            imagenet_val = imagenet_dataset(root_dir=self.root_dir, transform = self.transform_imagenet, num_class=self.num_class)      
            imagenet_loader = DataLoader(
                dataset=imagenet_val, 
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return imagenet_loader     

