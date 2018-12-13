# Traffic Light Classification

---

**Traffic Light Classification**

The goals/steps of this project are the following:

* Gather and label the datasets
* Transfer learning on a TensorFlow model
* Classify the state of traffic lights
* Summarize the results with a written report

[//]: # (References)
[capstone project]: https://github.com/RajeevSharma2015/Capstone-SDC
[Rajeev Sharma]: https://www.projectmanagement.com/profile/RajeevSharma1
[RajeevMachine-Setup]: ./imgs/RajeevMachine-Setup.jpg
[Tensorflow_GPU_Setup]: ./imgs/Tensorflow_GPU_Setup.jpg
[SSD V2 Train]: ./imgs/SSD-V2-Inception-20k.jpg
[Faster RCNN Train]: ./imgs/Faster-RCNN-20k.jpg
[Rajeev Ubuntu Setup]: #linux
[SSD Udacity 20K Steps]: https://drive.google.com/file/d/14O3IpuCKm1ZRdgeqk8RX3lSxCsCgsAN8/view?usp=sharing
[SSD Simulator 20K Steps]: https://drive.google.com/file/d/13ZexYyCWuM4XeIBsaa7bPN-G20a3wayv/view?usp=sharing
[FRCN 20k Steps]: https://drive.google.com/file/d/1hvMwgB4Im8UyybpdDchzNqPRru_4b3SJ/view?usp=sharing
[FRCN 10K Steps]: https://drive.google.com/file/d/1PLp_jA5yilzwK5il5tq-va7wo09HMWIT/view?usp=sharing


[bosch dataset]: https://hci.iwr.uni-heidelberg.de/node/6132
[lara dataset]: http://www.lara.prd.fr/benchmarks/trafficlightsrecognition
[alex lechner dataset]: https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0
[coldknight dataset]: https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset

[ssd inception]: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz 
[ssd inception 171117]: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
[faster rcnn inception]: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
[faster rcnn resnet101]: http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz

[coldknight repo]: https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI
[daniel stang]: https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e
[anthony sarkis]: https://codeburst.io/self-driving-cars-implementing-real-time-traffic-light-detection-and-classification-in-2017-7d9ae8df1c58
[vatsal srivastava]: https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62
[Alex's Lechner]: https://github.com/alex-lechner/Traffic-Light-Classification

[faster rcnn]: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
[labeling img]: ./imgs/labeling.jpg
[labelImg]: https://github.com/tzutalin/labelImg
[tf bad perfomance]: ./imgs/tf-bad-performance
[tfrecord file]: #23-create-a-tfrecord-file
[clifton pereira]: https://github.com/ExtEng
[ian burris]: https://github.com/iburris
[label map]: ./data/udacity_label_map.pbtxt
[set up tensorflow]: #set-up-tensorflow

[create_tf_record]: create_tf_record.py
[training section]: #training
[protobuf win]: https://github.com/google/protobuf/releases
[cdahms question]: https://stackoverflow.com/questions/48247921/tensorflow-object-detection-api-on-windows-error-modulenotfounderror-no-modu
[pythonpath win]: ./imgs/pythonpath-win.jpg
[path variable win]: ./imgs/path-win.jpg
[models zoo]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
[simultaneous training]: ./imgs/simultaneous-training.jpg

[bad performance]: ./imgs/tf-bad-performance.jpg
[model configs]: https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
[alex lechner model configs]: ./config
[7-zip win]: https://www.7-zip.org/
[aws login]: https://console.aws.amazon.com
[spot instance]: ./imgs/aws-spot-instance.jpg

[epratheeban github]: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/issues/11
[aws inbound rules]: ./imgs/aws-inbound-rules.jpg
[kill memory]: ./imgs/kill-process.jpg
[ssd udacity]: ./imgs/ssd-udacity-image.jpg
[ssd simulator]: ./imgs/ssd-simulator-image.jpg
[jupyter notebook]: ./tl_classification.ipynb


---

## Table of Contents
1. [Introduction](#introduction)
2. [Set up Tensorflow][set up tensorflow]
    1. [Rajeev Ubuntu Setup](#linux)
    2. [Ubuntu Tensorflow-Gpu][#linux-tf]
    3. [Windows 10](#windows-10)
3. [Datasets](#datasets)
    1. [The Lazy Approach](#1-the-lazy-approach)
    2. [The Diligent Approach](#2-the-diligent-approach)
        1. [Extract images from a ROSbag file](#21-extract-images-from-a-rosbag-file)
        2. [Data labeling](#22-data-labeling)
        3. [Create a TFRecord file][tfrecord file]
4. [Training][training section]
    1. [Choosing a model](#1-choosing-a-model)
    2. [Configure the .config file of the model](#2-configure-the-config-file-of-the-model)
    3. [Setup an AWS spot instance](#3-setup-an-aws-spot-instance)
    4. [Training the model](#4-training-the-model)
    5. [Freezing the graph](#5-freezing-the-graph)
5. [Recommendation: Models Performance](#Models Performance)
    1. [Conclusion](#conclusion)
6. [Troubleshooting](#troubleshooting)
7. [Summary](#summary)




## Introduction

The objective of this project was to retrain a TensorFlow model on images of traffic lights in their different light states (red, green, yellow & nolight). The trained model was then used in final " [capstone project] of Udacity SDC Nanodegree Program" as a frozen inference graph by IndiConcept team ( [Rajeev Sharma] & Malik ).

IndiConcept team project can be found here: [Drive Safely Capstone Project][capstone project]

The following guide is a detailed tutorial on how to set up the traffic light classification project, to (re)train the TensorFlow model, avoid common mistake and options available to achieve objective in different environments. 

As reference I've read [Daniel Stang's][daniel stang], [Anthony Sarkis'][anthony sarkis], [Vatsal Srivastava's][vatsal srivastava] and [Alex's Lechner] medium posts on traffic light classification. I encourage to read through them as well. 

Setting up a training environment is not easy in evolving OS/SW packages era. Moreover, retrain a Tensorflow model is a difficult task. This tutorial is Ubuntu friendly, and our training project completed on same. Still i tried to cover other working environment along with reference of other's work.

**Please check the [Troubleshooting section](#troubleshooting) for common issues**




## Set up TensorFlow
I will now show you how to install the TensorFlow 'models' repository on Windows 10 and Linux. I have done SSD and RCNN training on my local Ubuntu setup equipped with GPU. If you don't have a powerful GPU on your local machine I strongly recommend you to do the training on an AWS spot instance because this will save you a lot of time. 

However, you can do the basic stuff like data preparation and data preprocessing on your local machine but I suggest doing the training on an AWS instance. I will show you how to set up the training environment in the [Training section][training section].

#### Note
* Carla testing setup enabled by Tensorflow 1.3 version (Tensorflow 1.4 is Ok)
* So final inference graph need to be freeze on same version - if want to load and test on a compatible environment. 
* Meanwhile, a lot of incremental verson - CPU & GPU enabled released
* Tensorflw-GPU for python3.6 and ubuntu18.04 also work well for "object classification". 
* However train outcome of any other version need to freez on tensorflow==1.4 before launch on Carla car.
* Ubuntu setup and train instruction, herein are for advance versions of tensorflow-gpu (as per my machine)
* Window setup instruction guide us for tensorflow==1.4 [as per Alex tutorial]
* AWS setup guidence is based on Udacity AMI [as per Alex tutorial]




### My Linux (Ubuntu) Setup

My local machine is GPU enabled :
* Ubuntu - 18.04
* CUDA - 9.0
* CuDNN - 7.2
* Python - 3.6 
* Having seperate conda env for different tensorflow's is better approach i.e "tensorflow-gpu & tensorflow==1.4"


![RajeevMachine-Setup]


### Linux (ubuntu): "tensorflow-gpu & object classification" setup
1. Install TensorFlow-gpu version by executing 
    ```
    pip install tensorflow-gpu
    ```
 Herein you can verify installation by running a tensorflow test program.
 ![Tensorflow_GPU_Setup]
   
2. Install the following packages 
    ```
    sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
    ```
3. Create a new directory somewhere and name it ``tensorflow``
4. Clone TensorFlow's *models* repository from the ``tensorflow`` directory by executing 
    ```
    git clone https://github.com/tensorflow/models.git
    ```
5. Navigate to the ``models`` directory in the Command Prompt and execute 
    ```
    git checkout f7e99c0
    ```

    This is important because the code from the ``master`` branch won't work with TensorFlow version 1.4. Also, this commit has already fixed broken models from previous commits.

6.  Navigate to the ``research`` folder and execute 
    ```
    protoc object_detection/protos/*.proto --python_out=.

    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    ```
7.  If the step 6 executed without any errors then execute 
    ```
    python object_detection/builders/model_builder_test.py
    ```


#### Note
* Point 6 configuration always need to set before before using "Obect classification" library 
* This can be configured through script/setup or bashprofile



### Windows 10
Windows setup is for older version of tensorflow, also supported in Udacity program. In our project work we haven't done any activity on window's setup. This section is for convenience of student who wants to work in this environment.  

1. Install TensorFlow version 1.4 by executing the following statement in the Command Prompt (this assumes you have python.exe set in your PATH environment variable)
    ```
    pip install tensorflow==1.4
    ```
2. Install the following python packages 
    ```
    pip install pillow lxml matplotlib
    ```
3. [Download protoc-3.4.0-win32.zip from the Protobuf repository][protobuf win] (It must be version 3.4.0!)
4. Extract the Protobuf .zip file e.g. to ``C:\Program Files\protoc-3.4.0-win32``
5. Create a new directory somewhere and name it ``tensorflow``
6. Clone TensorFlow's *models* repository from the ``tensorflow`` directory by executing 
    ```
    git clone https://github.com/tensorflow/models.git
    ```
7. Navigate to the ``models`` directory in the Command Prompt and execute 
    ```
    git checkout f7e99c0
    ```

    This is important because the code from the ``master`` branch won't work with TensorFlow version 1.4. Also, this commit has already fixed broken models from previous commits.

8. Navigate to the ``research`` folder and execute 
    ```cmd
    ## The quotation marks are needed!
    “C:\Program Files\protoc-3.4.0-win32\bin\protoc.exe” object_detection/protos/*.proto --python_out=. 
    ```
9. If step 8 executed without any error then execute ``python builders/model_builder_test.py``
10. In order to access the modules from the research folder from anywhere, the ``models``, ``models/research``, ``models/research/slim`` & ``models/research/object_detection`` folders need to be set as PATH variables like so:

    10.1. Go to ``System`` -> ``Advanced system settings`` -> ``Environment Variables...`` -> ``New...`` -> name the variable ``PYTHONPATH`` and add the absolute path from the folders mentioned above

    ![pythonpath][pythonpath win]

    10.2. Double-click on the ``Path`` variable and add ``%PYTHONPATH%``

    ![path variable][path variable win]

Source: [cdahms' question/tutorial on Stackoverflow][cdahms question].


## Datasets
In deep learning, before you start coding you need to gather the right datasets. For this project, you will need images of traffic lights with labeled bounding boxes.

There are 4 datasets which can be used:
1. [Bosch Small Traffic Lights Dataset][bosch dataset]
2. [LaRA Traffic Lights Recognition Dataset][lara dataset]
3. Udacity's ROSbag file from Carla
4. Traffic lights from Udacity's simulator

#### Note:
Udacity SDC students ended up using Udacity's ROSbag file from Carla only and if you carefully follow along with this tutorial the images from the ROSbag file will be enough to have a working classifier for real-world AND simulator examples. There are two approaches on how to get the data from the ROSbag file (and from Udacity's simulator):

### 1. The Lazy Approach
You can download Vatsal Srivastava's dataset and my dataset for this project. The images are already labeled and a [TFRecord file][tfrecord file] is provided as well:

1. [Vatsal's dataset][coldknight dataset]
2. [My dataset][alex lechner dataset]

In our work we adopt lazy approach to avoid lengthy image processing steps and directly pick Vatsal/Alex pre-processes data. Both datasets include images from the ROSbag file and from the Udacity Simulator. I have used Vatsal's data for training and Alex for evaluation.

### 2. The Diligent Approach
If you have enough time, love to label images, read tutorials about traffic light classification before this one or want to gather more data, then this is the way to go:

#### 2.1 Extract images from a ROSbag file
For the simulator data, my team colleagues [Clifton Pereira][clifton pereira] and [Ian Burris][ian burris] drove around the track in the simulator and recorded a ROSbag file of their rides. Because Udacity provides the students with a ROSbag file from their Car named Carla where (our and) your capstone project will be tested on the code/procedure for extracting images will be (mostly) the same. **The steps below assume you have ros-kinetic installed either on your local machine (if you have Linux as an operating system) or in a virtual environment (if you have Windows or Mac as an operating system)**

1. Open a terminal and launch ROS 
    ```sh
    roscore
    ```
2. Open another terminal (but do NOT close or exit the first terminal!) and play the ROSbag file
    ```sh
    rosbag play -l path/to/your_rosbag_file.bag
    ```
3. Create a directory where you want to save the images
4. Open another, third terminal and navigate to the newly created directory and... 
    
    1. ...execute the following statement if you have a ROSbag file from Udacity's simulator:
        ```sh
        rosrun image_view image_saver _sec_per_frame:=0.01 image:=/image_color
        ```
    
    2. ...execute the following statement if you have a ROSbag file from Udacity's Car Carla:
        ```sh
        rosrun image_view image_saver _sec_per_frame:=0.01 image:=/image_raw
        ```

    As you can see the difference is the rostopic after ``image:=``.

These steps will extract the (camera) images from the ROSbag file into the folder where the code is executed. Please keep in mind that the ROSbag file is in an infinite loop and won't stop when the recording originally ended so it will automatically start from the beginning. If you think you have enough data you should interrupt one of the open terminals.

If you can't execute step 4.1 or 4.2 you probably don't have ``image_view`` installed. To fix this install ``image_view`` with ``sudo apt-get install ros-kinetic-image-view``.

Hint: You can see the recorded footage of your ROSbag file by opening another, fourth terminal and executing ``rviz``.

#### 2.2 Data labeling
After you have your dataset you will need to label it by hand. For this process I recommend you to [download labelImg][labelImg]. It's very user-friendly and easy to set up.
1. Open labelImg, click on ``Open Dir`` and select the folder of your traffic lights
2. Create a new folder within the traffic lights folder and name it ``labels``
3. In labelImg click on ``Change Save Dir`` and choose the newly created ``labels`` folder

Now you can start labeling your images. When you have labeled an image with a bounding box hit the ``Save`` button and the program will create a .xml file with a link to your labeled image and the coordinates of the bounding boxes.

Pro tip: I'd recommend you to split your traffic light images into 3 folders: Green, Yellow, and Red. The advantage is that you can check ``Use default label`` and use e.g. ``Red`` as an input for your red traffic light images and the program will automatically choose ``Red`` as your label for your drawn bounding boxes.

![labeling a traffic light][labeling img] 

#### 2.3 Create a TFRecord file
Now that you have your labeled images you will need to create a TFRecord file in order to retrain a TensorFlow model. A TFRecord is a binary file format which stores your images and ground truth annotations. But before you can create this file you will need the following:
1. A [``label_map.pbtxt``][label map] file which contains your labels (``Red``, ``Green``, ``Yellow`` & ``off``) with an ID (IDs must start at 1 instead of 0)
2. [Setup Tenorflow][set up tensorflow]
3. A script which creates the TFRecord file for you (feel free to use my [``create_tf_record.py``][create_tf_record] file for this process)

Please keep in mind that your ``label_map.pbtxt`` file can have more than 4 labels depending on your dataset. For example, if you're using the [Bosch Small Traffic Lights Dataset][bosch dataset] you will most likely have about 13 labels.

In case you are using the dataset from Bosch, all labels and bounding boxes are stored in a .yaml file instead of a .xml file. If you are developing your own script to create a TFRecord file you will have to take care of this. If you are using my script I will now explain how to execute it and what it does:

For datasets with **.yaml** files (e.g.: Bosch dataset) execute: 
```
python create_tf_record.py --data_dir=path/to/your/data.yaml --output_path=your/path/filename.record --label_map_path=path/to/your/label_map.pbtxt
```

For datasets with **.xml** files execute: 
```
python create_tf_record.py --data_dir=path/to/green/lights,path/to/red/lights,path/to/yellow/lights --annotations_dir=labels --output_path=your/path/filename.record --label_map_path=path/to/your/label_map.pbtxt
```

You will know that everything worked fine if your .record file has nearly the same size as the sum of the size of your images. Also, you have to execute this script for your training set, your validation set (if you have one) and your test set separately.

As you can see you don't need to specify the ``annotations_dir=`` flag for .yaml files because everything is already stored in the .yaml file.

The second code snippet (for datasets with .xml files) assumes you have the following folder structure:
```
path/to
|
└─green/lights
│   │  img01.jpg
│   │  img02.jpg
│   │  ...
|   |
│   └──labels
│      │   img01.xml
│      │   img02.xml
│      │   ...
|
└─red/lights
│   │  ...
|   |
│   └──labels
│      │   ...
|
└─yellow/lights
│   │  ...
|   |
│   └──labels
│      │   ...

``` 

## Training

### 1. Choosing a model
So far you should have a TFRecord file of the dataset(s) which you have either downloaded or created by yourself. Now it's time to select a model which you will train. You can [see the stats of and download the Tensorflow models from the model zoo][models zoo]. In sum I've trained 3 TensorFlow models and compared them based on their performance and precision:

* [SSD Inception V2 Coco (17/11/2017)][ssd inception 171117] Pro: Very fast, Con: Not good generalization on different data
* [SSD Inception V2 Coco (11/06/2017)][ssd inception] Pro: Very fast, Con: Not good generalization on different data
* [Faster RCNN Inception V2 Coco (28/01/2018)][faster rcnn inception] Pro: Good precision and generalization of different data, Con: Slow
* [Faster RCNN Resnet101 Coco (11/06/2017)][faster rcnn resnet101] Pro: Highly Accurate, Con: Very slow

Our team ended up using **SSD Inception V2 Coco (17/11/2017)** because it has good results for its performance.

#### Note:
You may ask yourself why the date after the model's name is important. As I've mentioned in the [TensorFlow set up section][set up tensorflow] above, it's very important to check out a specific commit from the 'models' repository because the team has fixed broken models. That's why it is important. And if you don't want to see the following results after a very long training session I encourage you to stick to the newest models or the ones I've linked above:

![bad performance][bad performance]

You get these result too if you have too few training steps. You can imagine how much time I've spent to figure this out...

After you've downloaded a model, create a new folder e.g. ``models`` and unpack the model with [7-zip on Windows][7-zip win] or ``tar -xvzf your_tensorflow_model.tar.gz`` on Linux.

### 2. Configure the .config file of the model
You will need to [download the .config file for the model you've chosen][model configs] or you can simply [download the .config files of this repository][alex lechner model configs] if you've decided to train the images on one of the models mentioned above.

If you want to configure them on your own there are some important changes you need to make. For this walkthrough, I will assume you are training on the Udacity Carla dataset with ~~Faster RCNN Inception V2~~ SSD Inception V2.

**TensorFlow model configs might differ but the following steps below are the same for every model!**

1. Change ``num_classes: 90`` to the number of labels in your ``label_map.pbtxt``. This will be ``num_classes: 4``
2. Set the default ``max_detections_per_class: 100`` and ``max_total_detections: 300`` values to a lower value for example ``max_detections_per_class: 10`` and ``max_total_detections: 10``
4. Change ``fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"`` to the directory where your downloaded model is stored e.g.: ``fine_tune_checkpoint: "models/your_tensorflow_model/model.ckpt"``
5. Set ``num_steps: 200000`` down to ``num_steps: 20000``
6. Change the ``PATH_TO_BE_CONFIGURED`` placeholders in ``input_path`` and ``label_map_path`` to your .record file(s) and ``label_map.pbtxt``

For Faster RCNN Inception V2:

1. Change the default ``min_dimension: 600`` and ``max_dimension: 1024`` values to the minimum value (height) and the maximum value (width) of your images like so

    ```
    keep_aspect_ratio_resizer {
        min_dimension: 1096
        max_dimension: 1368
    }
    ```

2. You can increase ``batch_size: 1`` to ``batch_size: 3`` or even higher (as per your machine capability)


If you don't want to use evaluation/validation in your training, simply remove those blocks from the config file. However, if you do use it make sure to set ``num_examples`` in the ``eval_config`` block to the sum of images in your .record file.

You can [take a look at the .config files of this repsoitory][alex lechner model configs] for reference. I've configured a few things like batch size and dropout as well. As I've mentioned earlier I've used [Vatsal's dataset][coldknight dataset] for training and my dataset for validation so don't get confused by the filename of my .record file ``jpg_udacity_train.record``.

### 3. Setup an AWS spot instance
For training, I recommend setting up an AWS spot instance. Training will be much faster and you can train multiple models simultaneously on different spot instances (like I did):

![simultaneous training][simultaneous training]
_Left: Training Faster RCNN Inception V2 Coco, Right: Training SSD Inception V2 Coco_

To set up an AWS spot instance do the following steps:
1. [Login to your Amazon AWS Account][aws login]
2. Navigate to ``EC2`` -> ``Instances`` -> ``Spot Requests`` -> ``Request Spot Instances``
3. Under ``AMI`` click on ``Search for AMI``, type ``udacity-carnd-advanced-deep-learning`` in the search field, choose ``Community AMIs`` from the drop-down and select the AMI (**This AMI is only available in US Regions so make sure you request a spot instance from there!**)
4. Delete the default instance type, click on ``Select`` and select the ``p2.xlarge`` instance
5. Uncheck the ``Delete`` checkbox under ``EBS Volumes`` so your progress is not deleted when the instance get's terminated
6. Set ``Security Groups`` to ``default``
7. Select your key pair under ``Key pair name`` (if you don't have one create a new key pair)
8. At the very bottom set ``Request valid until`` to about 5 - 6 hours and set ``Terminate instances at expiration`` as checked (You don't have to do this but keep in mind to receive a very large bill from AWS if you forget to terminate your spot instance because the default value for termination is set to 1 year.)
9. Click ``Launch``, wait until the instance is created and then connect to your instance via ssh

![spot instance][spot instance]

### 4. Training the model
1. When you're connected with the instance execute the following statements consecutively:
    ```sh
    sudo apt-get update
    pip install --upgrade dask
    pip install tensorflow-gpu==1.4
    ```

2. [Set up TensorFlow for Linux][tf setup linux] (**but skip step one because we've already installed tensorflow-gpu!**)
3. Clone your classification repository and create the folders ``models`` & ``data`` (in your project folder) if they are not tracked by your VCS.
4. Upload the datasets to the ``data`` folder 
    1. If you're using my dataset you can simply execute the following statements in the ``data`` folder: 

        ```sh
        wget https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0
        unzip alex-lechner-udacity-traffic-light-dataset.zip?dl=0 ## Don't miss the ``?dl=0`` part when unzipping!
        ``` 

5. Navigate to the ``models`` folder in your project folder and download your tensorflow model with 
    ```sh
    wget http://download.tensorflow.org/models/object_detection/your_tensorflow_model.tar.gz
    tar -xvzf your_tensorflow_model.tar.gz
    ```

6. Copy the file ``train.py`` from the ``tensorflow/models/research/object_detection`` folder to the root of your project folder
7. Train your model by executing the following statement in the root of your project folder
    ```
    python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=./config/your_tensorflow_model.config
    ```

#### SSD & RCNN Training Reference's (On Rajeev setup)
I have trained 4 different model's and tried in Capstone project:

* 1. SSD v2 inception coco simulation (20K steps)
* 2. SSD v2 inception coco udacity (20K steps)
* 3. Faster RCNN inception coco (10K steps)
* 4. Faster RCNN inception coco (20K steps)

Refence image for 20k SSD training completion - ![SSD V2 Train] 
Reference image for 20k Faster RCNN training completion - ![Faster RCNN Train]


### 5. Freezing the graph
When training is finished the trained model needs to be exported as a frozen inference graph. Udacity's Carla has TensorFlow Version 1.3 installed. However, the minimum version of TensorFlow needs to be Version 1.4 in order to freeze the graph but note that this does not raise any compatibility issues. 
If you've trained the graph with a higher version of TensorFlow than 1.4, don't panic! As long as you downgrade Tensorflow to version 1.4 before running the script to freeze the graph you should be fine.
To freeze the graph:
1. Copy ``export_inference_graph.py`` from the ``tensorflow/models/research/object_detection`` folder to the root of your project folder
2. Now freeze the graph by executing
    ```
    python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/your_tensorflow_model.config --trained_checkpoint_prefix ./models/train/model.ckpt-20000 --output_directory models
    ```

    This will freeze and output the graph as ``frozen_inference_graph.pb``.

### Conclusion
Our team is tested all 4 trained models however got better performance withd "SSD Inception V2 models". Roughly in 80% plus cases it do correct identification. 

Now we are using 2 models in our [Capstone project][capstone project]:
* 1 SSD model for real-world data
* 1 SSD model for simulator data

Here is Google Drive link for 4 trained models. 
* SSD Udacity 20K Steps 
* SSD Simulator 20K Steps 
* FRCN 20k Steps 
* FRCN 10K Steps 

**[Take a look at the Jupyter Notebook][jupyter notebook] to see the results.**


## Troubleshooting
In case you're running into any of the errors listed below, the solutions provided will fix it:

* Can't ssh into the AWS instance because of _port 22: Resource temporarily unavailable_

Go to ``Network & Security`` -> ``Security Groups`` -> right click on the security group that is used on your spot instance (propably ``default``) -> ``Edit inbound rules`` and set ``Source`` of SSH and Custom TCP to ``Custom`` and ``0.0.0.0/0`` like so:

![aws inbound rules][aws inbound rules]


* _tensorflow.python.framework.errors_impl.InternalError: Dst tensor is not initialized._

This error occurs when you don't have enough free available memory on your GPU to train. To fix this execute ``sudo fuser -v /dev/nvidia*`` and look for the process that is currently using your memory from the GPU.

![kill memory][kill memory]

Then kill the process by executing ``sudo kill -9 <PID-to-kill>``

* tensorflow-gpu training in between core dump [Faster RCNN inception V2 coco]
  You can decrease ``batch_size: 3`` to ``batch_size: 2`` or lower to 1. It depends how powerful GPU your local machine has. If it is Nvidia Geforce 1060 than i recommend to keep batch size: 1. A workable configuration file is available in ./config path

* tensorflow-gpu during training running out of resources [SSD inception V2 coco]
  You can decrease ``batch_size: 24`` to ``batch_size: 14`` or lower to 12. It depends how powerful GPU your local machine has. If it is Nvidia Geforce 1060 than i recommend to keep batch size: 14. A workable configuration file is available in ./config path   

## Summary
In IndiConcept team traffic classifier traning, we prefer to use earlier labeled dataset of Udacity. Since we lost a lot of time earlier in resolving platform dependencies so prefer to take a Lazy approach on Vatsal's and Alex's dataset:
1. [Download the datasets](#1-the-lazy-approach)
2. [Set up TensorFlow **only on the training instance**, do the training and export the model][training section]

There could be alternative approach of using own dataset, and step needs to follow :
1. [Set up TensorFlow locally][set up tensorflow] (because of creating TFRecord files)
2. [Create your own datasets](#2-the-diligent-approach)
3. [Set up TensorFlow again on a training instance (if the training instance is not your local machine), do the training and export the model][training section]


Training instance = System, where you train the TensorFlow model (probably an AWS instance and not your local machine)
