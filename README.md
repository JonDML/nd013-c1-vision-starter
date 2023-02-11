# Object Detection in an Urban Environment

## Table of contents
- [Object Detection in an Urban Environment](#object-detection-in-an-urban-environment)
  - [Table of contents](#table-of-contents)
  - [Project overview](#project-overview)
  - [Set up](#set-up)
    - [Build](#build)
    - [Open docker container terminal](#open-docker-container-terminal)
    - [Install gcloud](#install-gcloud)
    - [Download and Process Data](#download-and-process-data)
    - [Running 'Exploratory Data Analysis.ipynb'](#running-exploratory-data-analysisipynb)
    - [Model training and evaluation](#model-training-and-evaluation)
      - [Download pretrained models](#download-pretrained-models)
      - [Generate pipeline](#generate-pipeline)
  - [Dataset](#dataset)
    - [Dataset analysis](#dataset-analysis)
    - [Cross validation](#cross-validation)
  - [Training](#training)
    - [Reference experiment](#reference-experiment)
    - [Improve on the reference](#improve-on-the-reference)


## Project overview

In this project we are trying to achieve best performance possible to detect vehicles, pedrestians and cyclist. For this purpouse, we use Tensorflow API and [Waymo Open dataset](https://waymo.com/open/), giving a real data to work with. 
Object detection is one of the fundamental pillars in self driving cars, such as is necessary to know what they have around them.

## Set up
Once clonning this repository, we have to go to the build folder using command-line.  

### Build
Build image using:
```
docker build -t project-dev -f Dockerfile .
```

### Open docker container terminal
In the clonned directory, run the next command:
```
.\launch_docker.bat
```
This command will open the docker container terminal where we can work with the project.

### Install gcloud
Once in container, install gsutil:
```
curl https://sdk.cloud.google.com | bash
```
Once gsutil is installed, you can auth using:
```
gcloud auth login
```
### Download and Process Data
Inside container, we can run following script to download data and get files:
```
cd /app/project
python download_process.py --data_dir /app/project/data
```

### Running 'Exploratory Data Analysis.ipynb'
This can do easily from terminal:
```
./launch_jupyter.sh
```
This will give us a url to navigate (tested in chrome browser). Copy/paste given url, enter in **Exploratory Data Analysis.ipynb** file in browser, and run all commands.

### Model training and evaluation

This second part has been executed in remote computer due laptop limitations, so example paths will be changed than the previous commands.

#### Download pretrained models
Inside bash, paste following commands:
```
cd /home/workspace/experiments/pretrained_model/

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
```

#### Generate pipeline

Due to the problems to execute commands correctly, need to execute first this scripts to have a compatible version of packages

```
chmod +x /home/workspace/fix_packages_problems.sh
./home/workspace/fix_packages_problems.sh
```

Once finished, firefox doesn't crash and can explore jupyter notebooks. This fix break desktop **Visual Studio Code** link. So, to launch the editor, we need to execute:

```
chmod +x /home/workspace/launch_code.sh
./home/workspace/launch_code.sh
```

After this few steps, we can continue with work.

With following commands, we can generate pipeline to test later our data
```
cd /home/workspace/
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```

## Dataset

### Dataset analysis

Taking a look to images, can be explore that pedrestians are inside green bounding boxes, and cars between red bounding boxes, so, objects are classified correctly.

Also, can be observed in the following image, at background, there is a green bounding box which should not be there because inside we have not only cars, also pedrestians. So, the model is not perfect at classification work. Anyway, the model does a good job classificating because nearest objects are correctly classified and for self driving, this is the most important thing, due car need react first for nearest obstacles.

![One image capture](images/exploratory_data_analysis1.png)

Some multiple captures more to see results.

![Multiple images capture](images/exploratory_data_analysis_multiple.png)

This images are selected after looking some others because pedrestians and cars are mixed, so can be observed bounding boxes colors are correctly selected. This is city environment image, near a crosswalk, so there are more pedrestians than cars in images.
In other images, we can observe only cars, but can be used to check if algorithm works correctly.


### Cross validation

## Training

### Reference experiment

### Improve on the reference
