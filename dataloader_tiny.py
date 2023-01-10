from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch

# from vision import VisionDataset

from PIL import Image
from torchnet.meter import AUCMeter
import torch.nn.functional as F 
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from autoaugment import CIFAR10Policy, ImageNetPolicy
from tiny_pairflip_noise import *

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


transform_none_100_compose = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

transform_weak_100_compose = transforms.Compose(
    [
        transforms.RandomCrop(64),
        # transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.4, hue=0.07),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

transform_strong_100_compose = transforms.Compose(
    [
        transforms.RandomCrop(64),
        transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.4, hue=0.07),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    See :class:`DatasetFolder` for details.
    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)
    clsa, class_to_idx = find_classes(directory)
    # print(clsa,class_to_idx)
    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances,class_to_idx

class tiny_imagenet_dataset(Dataset):
    def __init__(self, SR, log, root, transform, mode, ratio, noise_mode, noise_file = '', num_samples=10000, pred=[], probability=[], paths=[], num_class=200):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.ratio = ratio
        self.noise_mode = noise_mode

        ### Get the instances and check if it is right
        data_folder = './data/tiny-imagenet-200/train/'
        train_instances, dict_classes = make_dataset(data_folder, extensions = IMG_EXTENSIONS)

        ## Validation Files
        data_folder = './data/tiny-imagenet-200/val/'
        val_instances = make_dataset(data_folder, extensions = IMG_EXTENSIONS)
        val_text = './data/tiny-imagenet-200/val/val_annotations.txt'
        val_img_files = './data/tiny-imagenet-200/val/images'
        num_class  = 200 
        num_sample = 100000
        data_folder = './data/tiny-imagenet-200/test/'
        test_instances = make_dataset(data_folder, extensions = IMG_EXTENSIONS)

        ## Load these instances->(data, label) into custom dataloader
        self.true_labels = {}
        self.test_labels  = {}
        self.val_labels   = {}
        self.train_labels = {}
        
        self.train_images = []
        self.val_imgs 	= []
        self.test_imgs 	= []


        for kk in range(len(train_instances)):
            path_ind = list(train_instances[kk])[0]
            self.true_labels[path_ind] =  int(list(train_instances[kk])[1])
            self.train_images.append(path_ind)

        # Get the Training Labels 
        train_label= []
        for kk in self.train_images:
            train_label.append(self.true_labels[kk])

        len_data = len(self.train_images)

        if os.path.exists(noise_file):
            noise_label = np.load(noise_file, allow_pickle=True)['label']     
            self.class_ind ={}
            for kk in range(num_class):
                self.class_ind[kk] = [i for i,x in enumerate(self.train_images) if noise_label.item()[x]==kk]

        else:                       ## Inject Noise
            noise_label = {}
            idx = self.train_images
            random.shuffle(idx)
            num_noise = int(ratio*len(idx))
            noise_idx = idx[:num_noise]
            noisy_index = []

            ## Check the Noise Type
            if noise_mode == 'instance_dependent':
                noise_label, actual_noise_rate, noise_idx = noisify_instance(self.train_images, self.true_labels, noise_rate=ratio)

            elif noise_mode=="asym":
                noiselabel, noise_rate = noisify('tiny_imagenet', num_class, np.array(train_label), 'pairflip', ratio, 0)
                num = 0
                for kk in self.train_images:
                    noise_label[kk] = noiselabel[num]
                    num += 1
            else:
                for i in idx:
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            noiselabel = random.randint(0,num_class-1)
                            noise_label[i] = noiselabel 

                        elif noise_mode == 'pair_flip':
                            noiselabel = self.pair_flipping[train_label[i]]
                            noise_label[i] = noiselabel

                        noisy_index.append(self.train_images.index(i))
                    else:
                        noise_label[i] = self.true_labels[i] 

            print("Save noisy labels to %s ..."%noise_file)
            np.savez(noise_file, label = noise_label, index = noisy_index, path = idx)
            idx = list(range(num_sample))
            random.shuffle(idx)
            clean_idx  = [x for x in idx if x not in noise_idx]       

            self.class_ind ={}
            for kk in range(num_class):
                self.class_ind[kk] = [i for i,x in enumerate(self.train_images) if noise_label[x]==kk]

        ## Indices for Clean Samples 
        save_file = "Tiny_ImageNet_" + str(noise_mode) + "_" +str(ratio) + ".npz" 

        ## For Warmup and JSD Calculation
        if self.mode == 'all':
            self.train_labels = noise_label
            self.train_imgs = self.train_images
            print("Number of Samples:", len(self.train_imgs))

        elif self.mode == "labeled":
            pred_idx = np.zeros(int(SR*num_sample))
            class_len = int(SR*num_sample/num_class)
            size_pred = 0

            ## Creating the Class Balance
            for i in range(num_class):
                class_indices = self.class_ind[i]
                prob1  = np.argsort(probability[class_indices].cpu().numpy())
                size1  = len(class_indices)
                try:
                    pred_idx[size_pred:size_pred+class_len] = np.array(class_indices)[prob1[0:class_len].astype(int)].squeeze()
                    size_pred += class_len
                except:                            
                    pred_idx[size_pred:size_pred+size1] = np.array(class_indices)
                    size_pred += size1

            ## Selected Clean Samples 
            pred_idx = [int(x) for x in list(pred_idx)]
            np.savez(save_file, index = pred_idx)
            self.train_imgs = np.array(self.train_images)[pred_idx]
            probability[probability<0.5] = 0
            self.probability = [1-probability[i] for i in pred_idx]
            print("%s data has a size of %d"%(self.mode, len(self.train_imgs)))
            self.train_labels = noise_label                 

        elif self.mode == "unlabeled":
            pred_idx       = np.load(save_file)['index']
            idx            = list(range(num_sample))
            pred_idx_noisy = [x for x in idx if x not in pred_idx]                                                   
            pred_idx = pred_idx_noisy   
            self.train_imgs = np.array(self.train_images)[pred_idx]
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))

        elif self.mode == 'val':
            with open(val_text,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = '%s/'%val_img_files+entry[0]
                    self.val_labels[img_path] = int(dict_classes[entry[1]])
                    self.val_imgs.append(img_path)



    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels.item()[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB') 

            ## Weakly and Strongly Augmeneted Copies 
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            return img1, img2, img3, img4, target, prob

        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')

            ## Weakly and Strongly Augmeneted Copies 
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4

        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels.item()[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index

        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)

class tinyImagenet_dataloader():  
    def __init__(self, root, batch_size, num_workers, log, ratio,  noise_mode, noise_file):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.ratio = ratio
        self.noise_mode = noise_mode
        self.log = log
        self.noise_file = noise_file

   #      self.transform_train = transforms.Compose([
			# 	transforms.RandomCrop(64),
			# 	transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.4, hue=0.07),
			# 	transforms.RandomHorizontalFlip(),		
			# 	transforms.ToTensor(),
			# ])

        self.transforms = {
            "warmup": transform_weak_100_compose,
            "unlabeled": [
                        transform_weak_100_compose,
                        transform_weak_100_compose,
                        transform_strong_100_compose,
                        transform_strong_100_compose
                    ],
            "labeled": [
                        transform_weak_100_compose,
                        transform_weak_100_compose,
                        transform_strong_100_compose,
                        transform_strong_100_compose
                    ],
            "test": None,
        }        

        self.transform_test = transforms.Compose([
				transforms.RandomCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])        

    def run(self,SR, mode,pred=[],prob=[],paths=[]):        
        if mode=='warmup':
            warmup_dataset = tiny_imagenet_dataset(SR, self.log ,self.root,transform=self.transforms["warmup"], mode='all', ratio = self.ratio, noise_mode = self.noise_mode, noise_file=self.noise_file)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*4,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader

        elif mode=='train':
            labeled_dataset = tiny_imagenet_dataset(SR, self.log ,self.root,transform=self.transforms["labeled"], mode='labeled',  ratio = self.ratio, noise_mode = self.noise_mode, noise_file=self.noise_file, pred=pred, probability=prob,paths=paths)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True, drop_last= True,
                num_workers=self.num_workers)

            unlabeled_dataset = tiny_imagenet_dataset(SR, self.log ,self.root,transform=self.transforms["unlabeled"], mode='unlabeled',  ratio = self.ratio, noise_mode = self.noise_mode, noise_file=self.noise_file, pred=pred, probability=prob,paths=paths)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True, drop_last= True,
                num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader

        elif mode=='eval_train':
            eval_dataset = tiny_imagenet_dataset(SR, self.log ,self.root,transform=self.transform_test, mode='all', ratio = self.ratio, noise_mode = self.noise_mode, noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=250,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        

        elif mode=='test':
            test_dataset = tiny_imagenet_dataset(SR, self.log ,self.root,transform=self.transform_test, mode='test', ratio = self.ratio, noise_mode = self.noise_mode, noise_file=self.noise_file)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=250,
                shuffle=False,
                num_workers=self.num_workers)             
            return test_loader             

        elif mode=='val':
            val_dataset = tiny_imagenet_dataset(SR, self.log ,self.root,transform=self.transform_test, mode='val', ratio = self.ratio, noise_mode = self.noise_mode, noise_file=self.noise_file)
            val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=250,
                shuffle=False,
                num_workers=self.num_workers)             
            return val_loader             
