# Single Image Crowd Counting via Multi Column Convolutional Neural Network

This is an unofficial implementation of CVPR 2016 paper ["Single Image Crowd Counting via Multi Column Convolutional Neural Network"](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)

# Installation
1. Install pytorch
2. Clone this repository
  ```Shell
  git clone https://github.com/svishwa/crowdcount-mcnn.git
  ```
  We'll call the directory that you cloned crowdcount-mcnn `ROOT`


# Data Setup
1. Download ShanghaiTech Dataset from   
   Dropbox:   https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0
   
   Baidu Disk: http://pan.baidu.com/s/1nuAYslz
2. Create Directory 
  ```Shell
  mkdir ROOT/data/original/shanghaitech/  
  ```
3. Save "part_A_final" under ROOT/data/original/shanghaitech/
4. Save "part_B_final" under ROOT/data/original/shanghaitech/
5. cd ROOT/data_preparation/

   run create_gt_test_set_shtech.m in matlab to create ground truth files for test data
6. cd ROOT/data_preparation/

   run create_training_set_shtech.m in matlab to create training and validataion set along with ground truth files

# Test
1. Follow steps 1,2,3,4 and 5 from Data Setup
2. Download pre-trained model files:

   [[Shanghai Tech A](https://www.dropbox.com/s/8bxwvr4cj4bh5d8/mcnn_shtechA_660.h5?dl=0)]
   
   [[Shanghai Tech B](https://www.dropbox.com/s/kqqkl0exfshsw8v/mcnn_shtechB_110.h5?dl=0)]
   
   Save the model files under ROOT/final_models
   
3. Run test.py

	a. Set save_output = True to save output density maps
	
	b. Errors are saved in  output directory

# Training
1. Follow steps 1,2,3,4 and 6 from Data Setup
2. Run train.py


# Training with TensorBoard
With the aid of [Crayon](https://github.com/torrvision/crayon),
we can access the visualisation power of TensorBoard for any 
deep learning framework.

To use the TensorBoard, install Crayon (https://github.com/torrvision/crayon)
and set `use_tensorboard = True` in `ROOT/train.py`.

# Other notes
1. During training, the best model is chosen using error on the validation set. (It is not clear how the authors in the original implementation choose the best model).
2. 10% of the training set is set asised for validation. The validation set is chosen randomly.
3. The ground truth density maps are obtained using simple gaussian maps unlike the original method described in the paper.
4. Following are the results on  Shanghai Tech A and B dataset:
		
                |     |  MAE  |   MSE  |
                ------------------------
                | A   |  110  |   169  |
                ------------------------
                | B   |   25  |    44  |
		
5. Also, please take a look at our new work on crowd counting using cascaded cnn and high-level prior (https://github.com/svishwa/crowdcount-cascaded-mtl),  which has improved results as compared to this work. 
               

