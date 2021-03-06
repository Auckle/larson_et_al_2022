
# Assessing Convolutional Neural Network Animal Classification Models for Practical Applications in Wildlife Conservation

All code associated with

Larson, Julia, "Assessing Convolutional Neural Network Animal Classification Models for Practical Applications in Wildlife Conservation" (2021). Master's Theses. 5184.
DOI: [https://doi.org/10.31979/etd.ysr5-th9v](https://doi.org/10.31979/etd.ysr5-th9v)
[https://scholarworks.sjsu.edu/etd_theses/5184](https://scholarworks.sjsu.edu/etd_theses/5184)

and any subsequent publications.

## Requirements and Installation

All scripts use Python 3+ with [Anaconda Individual Edition](https://www.anaconda.com/products/individual) for environment and package management. All scripts are run from the command line on a Linux system.

## Use

### 0. Setup a virtual machine - Optional

Machine learning projects are computationally heavy. If your personal machine or laptop is computationally light consider using a virtual machine (vm) from a cloud service such as [Microsoft's Azure](https://azure.microsoft.com/en-us/services/virtual-machines/#overview) ([tutorial](https://www.geeksforgeeks.org/azure-virtual-machine-for-machine-learning/)), [Amazon's AWS](https://aws.amazon.com/machine-learning/amis/) ([tutorial](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)), or [Google Cloud](https://cloud.google.com/deep-learning-vm) ([tutorial](https://medium.com/google-cloud/how-to-run-deep-learning-models-on-google-cloud-platform-in-6-steps-4950a57acfa5)). Virtual machines are available at reasonable prices preconfigured for Python machine learning applications using PyTorch and Anaconda. This will greatly decrease the time required to complete this project.

Basic instructions for setting up a vm using AWS follow.

#### A. Create an AWS account [here](https://aws.amazon.com/free/)

#### B. Launch a g4dn.xlarge [Deep Learning Base AMI (Ubuntu 18.04) Version 50.0 EC2 instance vm)](https://aws.amazon.com/marketplace/pp/prodview-x5nivojpquy6y) as described [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/launching-instance.html)

#### C. Create an 400 GiB gp2 empty Amazon EBS volume as shown [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-creating-volume.html)

#### D. Attach the EBS volume to the EC2 instance as show [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-attaching-volume.html)

#### E. Connect to the vm using ssh as described [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)

```bash
ssh -i [PATH_TO_KEY] ubuntu@[PUBLIC_DNS]
```

#### F. Make the EBS volume available for use as show [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html)

```bash
sudo mkdir /data
sudo chown -R ubuntu:ubuntu /data
 ```

 To determine the name of the EBS volume on your vm type the following command and look for the directory with the correct size that matches the created EBS volume.

```bash
lsblk
```

The first time the EBS volume is attached you must format the file system. This command will erase any files on the EBS volumn so only perform it once.

```bash
sudo mkfs -t xfs /dev/[VOLUME_NAME]
```

Mount the EBS volume to the data direcotry to make it availble to use.

```bash
sudo mount /dev/[VOLUME_NAME] /data
```

#### G. Update and Install Packages

To prepare the vm for running the coe use to following commands

```bash
sudo apt update
sudo apt upgrade
sudo apt-get -y install python3-pip
pip3 install humanfriendly jsonpickle tensorflow tensorflow-gpu==2.4 tqdm matplotlib
```

Skip to step 2 and in any subsequent steps follow any pip3/vm command options and ignore conda/local command options.

### 1. Install Anaconda

If not using a preconfigured machine, install Anaconda.

Download Anaconda [here](https://www.anaconda.com/products/individual) and follow the instructions for installing on you machine as explained [here](https://docs.anaconda.com/anaconda/install/).

A getting started guide for using conda is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

### 2. Clone this Repo

#### A. Clone this repo in to your workspace

conda/local & pip3/vm:

```bash
git clone https://github.com/Auckle/larson_et_al_2022.git
```

#### B. Change directory into the cloned repo

conda/local & pip3/vm:

```bash
cd [YOUR_WORKSPACE]/larson_et_al_2022
```

### 3. Download the Wellington Camera Traps Dataset

Retrieve the [Wellington Camera Traps](https://lila.science/datasets/wellingtoncameratraps) dataset and metadata from the Labeled Information Library of Alexandria: Biology and Conservation (LILA BC) [website](https://lila.science/).

#### A. 1. Download and unzip the Wellington Camera Traps images into the data/raw directory

conda/local & pip3/vm:

```bash
wget -c -P ./data/raw https://lilablobssc.blob.core.windows.net/wellingtoncameratraps/wct_images.zip
```

pip3/vm:

```bash
unzip ./data/raw/wct_images.zip -d ./data/raw
```

or, if difficulties are encountered in downloading the zip file because of its size:

#### A. 2. a. Download the azcopy command-line utility

conda/local:
Download the azcopy command-line utility as described [here](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10)

pip3/vm:

```bash
wget -c -P . https://aka.ms/downloadazcopy-v10-linux
tar -xvzf  downloadazcopy-v10-linux
```

#### A. 2. b. Use the azopy util to download images one at a time to the data/raw directory

conda/local:

```bash
azcopy cp "https://lilablobssc.blob.core.windows.net/wellington-unzipped/images/*" "./data/raw" --recursive --overwrite=false
```

pip3/vm:

```bash
 ./azcopy_linux_amd64_10.14.1/azcopy cp "https://lilablobssc.blob.core.windows.net/wellington-unzipped/images/*" "./data/raw" --recursive --overwrite=false
```

#### B. Download and unzip the Wellington Camera Traps metadata .csv file into the data/raw directory

conda/local:

```bash
wget -c -P ./data/raw https://lilablobssc.blob.core.windows.net/wellingtoncameratraps/wellington_camera_traps.csv.zip
```

pip3/vm:

```bash
unzip ./data/raw/wellington_camera_traps.csv.zip -d ./data/raw
```

### 4. Run Microsoft's Megadetector

Download and use [Microsoft's Megadetector model](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md) to detect animals in the Wellington Camera Traps images, and generate an output file containing the boundary box coordinates of detected animals.

#### A. Download the megadetector.pb model to the megadetector directory

conda/local & pip3/vm:

```bash
wget -c -P ./megadetector https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb
```

#### B. Clone Microsoft's cameratraps and ai4eutils repos

conda/local & pip3/vm:

```bash
git clone https://github.com/Microsoft/cameratraps
git clone https://github.com/Microsoft/ai4eutils
```

#### C. Add Microsoft's repos to your Python path

conda/local & pip3/vm:

```bash
export PYTHONPATH=$PYTHONPATH:$PWD/cameratraps
export PYTHONPATH=$PYTHONPATH:$PWD/ai4eutils
```

#### D. If using a vm skip to Step G, otherwise create the conda virtual environment from microsoft's .yml file. See [here](https://github.com/microsoft/CameraTraps#installation) for more

conda/local:

```bash
conda env create --file cameratraps/environment-detector.yml
```

#### E. Activate the conda environment. See [here](https://github.com/microsoft/CameraTraps#installation) for more

conda/local:

```bash
conda activate cameratraps-detector
```

#### F. Install additional packages into the Python environment

conda/local:

```bash
conda install pandas pytorch torchvision
```

G. Run Microsoft's run_tf_detector_batch.py script to use the Megadetector on the raw Wellington Camera Traps images and generate the detections output file detections.json. See [2. run_tf_detector_batch.py](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md#2-run_tf_detector_batchpy) for more.

conda/local & pip3/vm:

```bash
python3 ./cameratraps/detection/run_tf_detector_batch.py ./megadetector/megadetector_v3.pb ./data/raw ./megadetector/detections.json

python3 ./cameratraps/detection/run_tf_detector_batch.py ./megadetector/megadetector_v3.pb ./data/raw ./megadetector/detections.json --checkpoint_frequency 100
```

More detailed instructions and trouble shooting for using the megadetector can be found [here](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md) and [here](https://github.com/microsoft/CameraTraps#installation).

### 5. Prepare Classification Training and Testing Datasets

Generate and process the classification dataset images, sort them into training and testing datasets, and clean the metadata.

#### A. Generate the classification dataset images by running the crop_detections.py script to crop the animals from the raw images using the detections generated in step 3. Run the Microsoft's Megadetector. Images labeled as empty according to the Wellington Camera Traps metadata file are cropped using the boundary box coordinates of the previous detection

conda/local & pip3/vm:

```bash
python3 crop_detections.py ./megadetector/detections.json ./data/raw/wellington_camera_traps.csv ./data/raw ./data/detections
```

#### B. Sort the cropped images into ten training and seven testing datasets using the prepare_classification_datasets.py script

conda/local & pip3/vm:

```bash
python3 prepare_classification_datasets.py --detections_csv=./data/detections/wellington_camera_traps_detections.csv --output_csv=training_testing_datasets.csv
```

This script also:

- Combines images labeled as rat, Norway rat, and ship rat into a single "rat" label.
- Combines images labeled as hare and rabbit into a single "hare" label.
- Changes images labeled as nothinghere to "empty."
- Generates two labels for each image:
  - An individual label specifying the species or group of species.
  - A grouped label where birds are labeled as "native," empty frames are labeled as "empty," and all others (unclassifiable, cat, deer, dog, hare, hedgehog, mouse, mustelid, pig, possum, and rat) are labeled as "invasive."

### 6. CNN Training

Train the ten convolutional neural network models using each of the ten training datasets with the training.py script.

pip3 install torch torchvision

Use the following command to start training fresh.
conda/local:

```bash
python3 training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections
```

pip3/vm:

```bash
python3 training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections --num_workers=4
```

Use the following command to resume training after stopping from saved checkpoints.
conda/local:

```bash
python3 training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections --resume_training
```

pip3/vm:

```bash
python3 training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections --resume_training --num_workers=4
```

Each model uses the ResNet-18 architecture pretrained on the imagenet database with a rectified linear activation function (ReLU), Stochastic Gradient Descent with momentum for backpropagation, and learning rates and weight decays matching those of [Norouzzadeh et al. 2018](https://doi.org/10.1073/pnas.1719367115) and [Tabak et al. 2018](https://doi.org/10.1111/2041-210X.13120).

Standard data augmentation techniques (random horizontal flipping, cropping, and color jitter) are applied.

### 7. CNN Testing and Evaluation

Test and evaluate the ten models using the seven testing datasets in 13 combinations using the training.py script with the --evaluate flag.
conda/local:

```bash
python3 training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections --evaluate
```

pip3/vm:

```bash
python3 training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections --evaluate --num_workers=4
```

This script generates predictions for each train/test combination and stores them in a .csv file in the training/predictions directory and a single file called model_performance_results.csv containing the following performance metrics across each train/test combination, and by site and class/label within each train/test combination:

- Top-1 accuracy: # of times the top prediction is correct / # of images tested
- Top-5 accuracy: # of times the correct class is in the top 5 predictions / # of images tested (models with five or fewer output classes will always have top-5 accuracies of 100%)
- False alarm rate: # of empty or native images labeled as invasive / # of images tested
- Missed invasive rate: # of invasive images labeled as native or empty / # of invasive images tested

### 8. Figures

Generate the paper figures with the generate_figures.py script.

#### A. Install additional packages into the Python environment

conda/local:

```bash
conda install bokeh seaborn
```

pip3/vm:

```bash
pip3 install bokeh seaborn
```

#### B. Generate figures

conda/local & pip3/vm:

```bash
python3 generate_figures.py --v_thesis
```
