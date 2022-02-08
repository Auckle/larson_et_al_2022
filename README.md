
# Article Title TODO

Abstract and Introduction [[1]](#references)

## Requirements and Installation TODO

All scripts, training, and analysis are run in Python using Jupyter notebooks, with Python environments and packaged managed via [Anaconda](https://www.anaconda.com/products/individual)

Python version 3+

Packages:
- Keras with TensorFlow backend
- NumPy
- also install Pandas
- sklearn
- torch
- cameratrap megadetector: tensorflow pillow humanfriendly matplotlib tqdm jsonpickle statistics requests
tensorflow pandas tqdm pillow humanfriendly matplotlib tqdm jsonpickle statistics requests
conda install pytorch torchvision torchaudio -c pytorch

#

## Use TODO

### 

### 1. clone this repo to your computer

Consists of links to other .md file 

### 2. Download the Camera-Trap Dataset 

You can just do Pseudocode for data preparation if wanted

- Download and unzip the images, and the csv file inot the data/raw directory
https://lila.science/datasets/wellingtoncameratraps, into the data/raw/ folder.
- convert download to json here? http://localhost:8888/notebooks/Desktop/journal_article/thesis_work/archive/jl_run_b_wellington_to_json.ipynb 

### Megadetector (Training and Testing Datasets )
Use Microsoft's Megadector model to detect animals in the Welling Cameratrap database images and generate an output file containing the boundary box coordinates of the animal, if any, in each image.

1. Using the terminal move into the megadetector directory containing the script TODO which script?
```cd into megadector```

2. Follow step 0. prerequisites the [instructions here](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md) were follwed to setup your computer and environment, download and run the megadetector  using option

- for step 2. = Download the megadetector.pb model to megadetector director I used version MegaDetector v3, 2019.05.30

- cd diretory into larson_et_al_2022
- for step 4. = 
```export PYTHONPATH=$PYTHONPATH:$PWD/megadetector/cameratraps``` and
```export PYTHONPATH=$PYTHONPATH:$PWD/megadetector/ai4eutils```

2.1
```conda env create --file cameratraps/environment-detector.yml```
2.2 activate the enviornment
```conda activate cameratraps-detector```

3. Follow the steps to run 2. run_tf_detector_batch.py
```python ./megadetector/cameratraps/detection/run_tf_detector_batch.py ./megadetector/megadetector_v3.pb ./data/raw ./megadetector/detections.json```

### Image Cropping
Use the detections from the Megadetector to crop the images and build the dataset for use in training the images classfiers using the crop_detections.py script.
```python crop_detections.py ./megadetector/detections.json ./data/raw/wellington_camera_traps.csv ./data/raw ./data/detections```

### Prepare Classification Dataset - (Methods -  Camera-trap Image Dataset, Training and Testing Datasets)
Using the cropped images generate ten training and seven testing datasets using the prepare_classification_datasets.py script
```python prepare_classification_datasets.py --detections_csv=./data/detections/wellington_camera_traps_detections.csv```

This script cleans up the dataset by " Images labeled as rat, Norway rat, and ship rat were combined in a single "rat" label, hare and rabbit were combined into a single "hare" label. Each image was given two labels for different models—an individual label specifying the species or group of species and a grouped label where birds were labeled as "native," empty frames labeled as "empty," and all others (unclassifiable, cat, deer, dog, hare, hedgehog, mouse, mustelid, pig, possum, and rat) labeled as "invasive." "

And this script sorts the datasets according the graphic below.
TODO insert Figure 6 updated, Table 3, and Table 4


run
```python prepare_classification_datasets.py --detections_csv=./data/detections/wellington_camera_traps_detections.csv --output_csv=training/training_testing_datasets.csv
```


 TODO remove below: 
 this is for testing:
 python prepare_classification_datasets.py --detections_csv=./data/raw/wellington_camera_traps.csv
 put line ```db['org_file'] = db['file']``` at line 276


### 2. Training (sections Model Development and Training )
"To increase the size of the training datasets and to increase CNN accuracy, standard data augmentation techniques were applied to all training images (Shorten et al., 2019). Augmentation included random horizontal flipping, cropping, and color jitter."

use the following to start training fresh.
```
python training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections
```

use the follow to resume training after stopping from saved checkpoints
```
python training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections --resume_training
```



### 3. Testing
```
python training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections --evaluate
```


### 4. Results

to generate figures run 
```
python generate_figures.py
```
#

## Acknowledgement TODO

#

## References TODO

1. Example Reference
