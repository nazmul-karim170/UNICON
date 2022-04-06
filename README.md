# UNICON-Noisy-Label
Official Implementation of the CVPR 2022 paper "UNICON: Combating Label Noise Through Uniform Selection and Contrastive Learning"
https://arxiv.org/pdf/2203.14542.pdf

<!-- ![Teaser](./Figure/Teaser.png) -->
![Framework](./Figure/Snip20220331_3.png)

# Example Run
After creating a virtual environment, run 'pip install -r requirements.txt'	
Example run (CIFAR10 with 50% symmetric noise) 

	python Train_cifar.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.5 

Example run (CIFAR100 with 90% symmetric noise) 

	python Train_cifar.py --dataset cifar100 --num_class 100 --data_path ./data/cifar100 --noise_mode 'sym' --r 0.9 
	
This will throw an error as downloaded files will not be in proper folder. That is why they are needed to be manually moved to the "data_path".

# Dataset
For datasets other than CIFAR10 and CIFAR100, you need to download them from their corresponsing website.

# Reference 
If you have any questions, do not hesitate to contact at nazmul.karim18@knights.ucf.edu

Also, if you find our work useful please cite: 

@misc{https://doi.org/10.48550/arxiv.2203.14542,
  doi = {10.48550/ARXIV.2203.14542},
  url = {https://arxiv.org/abs/2203.14542},
  author = {Karim, Nazmul and Rizve, Mamshad Nayeem and Rahnavard, Nazanin and Mian, Ajmal and Shah, Mubarak},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {UNICON: Combating Label Noise Through Uniform Selection and Contrastive Learning},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
 
