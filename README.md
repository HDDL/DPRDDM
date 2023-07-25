# DPRDDM
## Doubly physics-regularized denoising diffusion model


Implemented by Zhenjie Zheng, advised by Wei Ma, Civil and environmental engineering, The Hong Kong Polytechnic University. 


### Requirements

- Python 3.9
- Pytorch
- Numpy
- einops
- PIL
- ema_pytorch
- accelerate

### Introductions

The DPRDDM is an unsupervised deep learning model used to remove the corrupted noises in traffic speed data.  The input only includes the noisy speed matrix with full observation. Then, the DPRDDM outputs the recovered speed matrix.

### Instructions
You can directly run the DPRDDM_main to train the DPRDDM and recovered speed matrices will be saved. The origin MAPE, origin RMSE, reocvered MAPE, and recovered RMSE will be reported. You can also demonstrate the corresponding speed matrices with figures to visually observe the denoising performance of our model.

### Folders of repo
The repo contains two folders: src and data.

#### src
src contains the code of DPRDDM that is implemented with Python using Zen Traffic data. It takes a long time to run once, and 10 denoising models will be saved during the training. You can use these models to remove the noises in traffic speed data with the function trainer.train_data_remove_noise(). The recovered speed matrices will be saved in the results folder. 

#### data

We use the Zen Traffic data (https://zen-traffic-data.net/english/) to train the model. Since the Zen Traffic data can only be available with the permission of Hanshin Expressway Company, we cannot provide the data directly. However, we provide partial demo datasets in the data folder to demonstrate the performance of the DPRDDM. To obtain all speed matrices used in the DPRDDM, please apply the Zen Traffic data first (https://zen-traffic-data.net/english/description/). Then, we will send you the associated noisy speed matrices by email. Our email address is zhengzj17@gmail.com



### Contact
For any question, please contact zhengzj17@gmail.com
