# Semantic Segmentation

### Goal
The goal of the project is to use the KITTY road dataset to build a fully connected convolutional network to identify
drivable road sections using a pretrained VGG-16 network.

### Model
The pre-trained VGG-16 network was converted to fully connected after detaching the final fully connected layer and
replacing it with 1x1 convolution to preserve spatial information. For layers 3, 4, skip connections is applied to
improve performance (1x1 convolutions are applied and the result is added element wise with the transposed convolution
of the previous layer to upsample the image to its original size). All convolutions (1x1 and transposed) are using
regularization.

### Optimization
Since this is a classification problem, the loss function was set to cross entropy. Adam optimizer was used.

### Parameters
keep_probability_value = 0.5
learning_rate_value = 1e-4
num_of_epochs = 20

Using my own graphics card (970/4GB RAM), i was unable to fit the model in memory, even after setting:
batch_size = 2

I used the service at https://www.floydhub.com/, and let batch_size=2 to be sure that the model would be trained.
This was true, but given more powerups, i would increase the value up to 16 to check what happens.

### Output

```
2017-12-04 15:19:07,017 INFO - Preparing to run TaskInstance <TaskInstance: antalakas/projects/carnd-term3-p2/13 (id: QwKmhbx92umiYWx5kADVYP)
2017-12-04 15:19:07,028 INFO - Starting attempt 1 at 2017-12-04 23:19:07.021160
2017-12-04 15:19:07,037 INFO - Downloading and setting up data sources
2017-12-04 15:19:07,045 INFO - Downloading and mounting dataset. ETA: 32 seconds
2017-12-04 15:19:30,907 INFO - Pulling Docker image: floydhub/tensorflow:1.3.1-gpu-py3_aws.13
2017-12-04 15:19:32,143 INFO - Starting container...
2017-12-04 15:19:32,305 INFO -
################################################################################

2017-12-04 15:19:32,305 INFO - Run Output:
2017-12-04 15:19:33,440 INFO - TensorFlow Version: 1.3.1
2017-12-04 15:19:33,511 INFO - 2017-12-04 23:19:33.510132: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-12-04 15:19:33,511 INFO - 2017-12-04 23:19:33.510469: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties:
2017-12-04 15:19:33,511 INFO - name: Tesla K80
2017-12-04 15:19:33,511 INFO - major: 3 minor: 7 memoryClockRate (GHz) 0.8755
2017-12-04 15:19:33,512 INFO - pciBusID 0000:00:1e.0
2017-12-04 15:19:33,512 INFO - Total memory: 11.17GiB
2017-12-04 15:19:33,512 INFO - Free memory: 11.10GiB
2017-12-04 15:19:33,512 INFO - 2017-12-04 23:19:33.510502: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0
2017-12-04 15:19:33,512 INFO - 2017-12-04 23:19:33.510521: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y
2017-12-04 15:19:33,512 INFO - 2017-12-04 23:19:33.510546: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0)
2017-12-04 15:19:33,643 INFO - 2017-12-04 23:19:33.643142: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0)
2017-12-04 15:19:33,644 INFO - Default GPU Device: /gpu:0
2017-12-04 15:19:33,645 INFO - 2017-12-04 23:19:33.644832: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0)
2017-12-04 15:19:33,662 INFO - Tests Passed
2017-12-04 15:19:39,876 INFO - Tests Passed
2017-12-04 15:19:39,928 INFO - 2017-12-04 23:19:39.927642: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0)
2017-12-04 15:19:40,160 INFO - Tests Passed
2017-12-04 15:19:40,163 INFO - 2017-12-04 23:19:40.162499: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0)
2017-12-04 15:19:40,166 INFO - Tests Passed
2017-12-04 15:19:40,170 INFO - Tests Passed
2017-12-04 15:19:40,170 INFO - 2017-12-04 23:19:40.170262: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0)
2017-12-04 15:21:28,434 INFO - Epoch: 1 / 20  Loss: 0.799  Time:  0:01:40.796693
2017-12-04 15:23:06,229 INFO - Epoch: 2 / 20  Loss: 0.685  Time:  0:01:37.794662
2017-12-04 15:24:44,032 INFO - Epoch: 3 / 20  Loss: 0.626  Time:  0:01:37.801460
2017-12-04 15:26:21,925 INFO - Epoch: 4 / 20  Loss: 0.489  Time:  0:01:37.894102
2017-12-04 15:27:59,764 INFO - Epoch: 5 / 20  Loss: 0.142  Time:  0:01:37.839469
2017-12-04 15:29:37,561 INFO - Epoch: 6 / 20  Loss: 0.116  Time:  0:01:37.796616
2017-12-04 15:31:15,419 INFO - Epoch: 7 / 20  Loss: 0.088  Time:  0:01:37.857929
2017-12-04 15:32:53,233 INFO - Epoch: 8 / 20  Loss: 0.268  Time:  0:01:37.814489
2017-12-04 15:34:31,108 INFO - Epoch: 9 / 20  Loss: 0.148  Time:  0:01:37.874265
2017-12-04 15:36:08,918 INFO - Epoch: 10 / 20  Loss: 0.341  Time:  0:01:37.810612
2017-12-04 15:37:46,749 INFO - Epoch: 11 / 20  Loss: 0.089  Time:  0:01:37.830616
2017-12-04 15:39:24,557 INFO - Epoch: 12 / 20  Loss: 0.053  Time:  0:01:37.808496
2017-12-04 15:41:02,355 INFO - Epoch: 13 / 20  Loss: 0.091  Time:  0:01:37.797544
2017-12-04 15:42:40,194 INFO - Epoch: 14 / 20  Loss: 0.038  Time:  0:01:37.838247
2017-12-04 15:44:18,071 INFO - Epoch: 15 / 20  Loss: 0.060  Time:  0:01:37.877882
2017-12-04 15:45:55,922 INFO - Epoch: 16 / 20  Loss: 0.113  Time:  0:01:37.850372
2017-12-04 15:47:33,747 INFO - Epoch: 17 / 20  Loss: 0.139  Time:  0:01:37.824820
2017-12-04 15:49:11,572 INFO - Epoch: 18 / 20  Loss: 0.046  Time:  0:01:37.825582
2017-12-04 15:50:49,398 INFO - Epoch: 19 / 20  Loss: 0.036  Time:  0:01:37.825835
2017-12-04 15:52:27,250 INFO - Epoch: 20 / 20  Loss: 0.084  Time:  0:01:37.851597
2017-12-04 15:52:27,252 INFO - Training Finished. Saving test images to: /output/runs/1512431547.2492785
2017-12-04 15:55:36,836 INFO -
################################################################################

2017-12-04 15:55:36,838 INFO - Waiting for container to complete...
2017-12-04 15:55:37,079 INFO - Persisting outputs...
2017-12-04 15:55:38,669 INFO - [success] Finishing execution in 2191 seconds for TaskInstance <TaskInstance: antalakas/projects/carnd-term3-p2/13 (id: QwKmhbx92umiYWx5kADVYP)
2017-12-04 15:55:38,677 INFO - Creating data module for output...
2017-12-04 15:55:38,705 INFO - Data module created for output.
```


### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
