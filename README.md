# UNICON-Noisy-Label
Official Implementation of the CVPR 2022 paper "UNICON: Combating Label Noise Through Uniform Selection and Contrastive Learning"
![Link to Paper][https://arxiv.org/pdf/2203.14542.pdf]

<!-- ![Teaser](./Figure/Teaser.png) -->
![Framework](./Figure/Snip20220331_3.png)

After creating a virtual environment, run 'pip install -r requirements.txt'

	
Example run (CIFAR10 with 50% symmetric noise) 

	python Train_cifar.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.5 

Example run (CIFAR100 with 90% symmetric noise) 

	python Train_cifar.py --dataset cifar100 --num_class 100 --data_path ./data/cifar100 --noise_mode 'sym' --r 0.9 
	
This will throw an error as downloaded files will not be in proper folder. That is why they need to be manually moved to the "data_path".

For datasets other than CIFAR10 and CIFAR100, you need to manually download them. 
 
 
