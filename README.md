<h2 align="center"> <a href="https://github.com/nazmul-karim170/UNICON-Noisy-Label">UNICON: Combating Label Noise Through Uniform Selection and Contrastive
Learning</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2312.09313-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2203.14542.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/nazmul-karim170/UNICON-Noisy-Label/blob/main/LICENSE) 


</h5>

## [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Karim_UniCon_Combating_Label_Noise_Through_Uniform_Selection_and_Contrastive_Learning_CVPR_2022_paper.pdf) 


## Code for Training 


### UNICON Framework

<!-- ![Teaser](./Figure/Teaser.png) -->
![Framework](./Figure/Snip20220331_3.png)

### Example Run
After creating a virtual environment, run

	pip install -r requirements.txt

Example run (CIFAR10 with 50% symmetric noise) 

	python Train_cifar.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.5 

Example run (CIFAR100 with 90% symmetric noise) 

	python Train_cifar.py --dataset cifar100 --num_class 100 --data_path ./data/cifar100 --noise_mode 'sym' --r 0.9 

This will throw an error as downloaded files will not be in the proper folder. That is why they must be manually moved to the "data_path".

Example Run (TinyImageNet with 50% symmetric noise)

	python Train_TinyImageNet.py --ratio 0.5


Example run (Clothing1M)
	
	python Train_clothing1M.py --batch_size 32 --num_epochs 200   

Example run (Webvision)
	
	python Train_webvision.py 


### Dataset
For datasets other than CIFAR10 and CIFAR100, you need to download them from their corresponding website.

### Reference 
If you have any questions, do not hesitate to contact nazmul.karim170@gmail.com

Also, if you find our work useful please consider citing our work: 

	@InProceedings{Karim_2022_CVPR,
	    author    = {Karim, Nazmul and Rizve, Mamshad Nayeem and Rahnavard, Nazanin and Mian, Ajmal and Shah, Mubarak},
	    title     = {UniCon: Combating Label Noise Through Uniform Selection and Contrastive Learning},
	    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	    month     = {June},
	    year      = {2022},
	    pages     = {9676-9686}
	}
 
