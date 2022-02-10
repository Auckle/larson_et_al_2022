
# Assessing Convolutional Neural Network Animal Classification Models for Practical Applications in Wildlife Conservation

All code associated with:

Larson, Julia, "Assessing Convolutional Neural Network Animal Classification Models for Practical Applications in Wildlife Conservation" (2021). Master's Theses. 5184.
DOI: [https://doi.org/10.31979/etd.ysr5-th9v](https://doi.org/10.31979/etd.ysr5-th9v)
[https://scholarworks.sjsu.edu/etd_theses/5184](https://scholarworks.sjsu.edu/etd_theses/5184)

and any subsequent publications.


## Requirements and Installation
All scripts use Python 3+ with [Anaconda Individual Edition](https://www.anaconda.com/products/individual) for environment and package management. All scripts are run from the command line on a Linux system.

## Use

### 1. Clone this Repo
1. Clone this repo in to your workspace
```
git clone 
```
2. Change directory into the cloned repo.
```
cd [YOUR_WORKSPACE]/larson_et_al_2022
```

### 2. Download the Wellington Camera Traps Dataset
Retrieve the [Wellington Camera Traps](https://lila.science/datasets/wellingtoncameratraps)[[1]](#references) dataset and metadata from the Labeled Information Library of Alexandria: Biology and Conservation ([ILA BC](https://lila.science/)) website.
1. Download and unzip the [Wellington Camera Traps](https://lila.science/datasets/wellingtoncameratraps) images into the data/raw directory.
```
wget -c -P ./data/raw https://lilablobssc.blob.core.windows.net/wellingtoncameratraps/wct_images.zip
```
2. Download and unzip the [Wellington Camera Traps](https://lila.science/datasets/wellingtoncameratraps) metadata .csv file into the data/raw directory.
```
wget -c -P ./data/raw https://lilablobssc.blob.core.windows.net/wellingtoncameratraps/wellington_camera_traps.csv.zip
```

### 3. Run the Microsoft's Megadetector
Download and use [Microsoft's Megadetector model]((https://github.com/microsoft/CameraTraps/blob/main/megadetector.md))[[2]](#references) to detect animals in the Wellington Camera Traps images, generate an output file containing the boundary box coordinates of detected animals, and create the classification dataset by using the boundary box coordinates to crop the animals from the original images.

1. Download the megadetector.pb model to the megadetector directory
```
wget -c -P ./megadetector https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb
```

2. Clone Microsoft's cameratraps and ai4eutils repos into the megadetector directory.
```
git clone https://github.com/Microsoft/cameratraps megadetector
git clone https://github.com/Microsoft/ai4eutils megadetector`
```

3. Add Microsoft's repos to your Python path
```
export PYTHONPATH=$PYTHONPATH:$PWD/megadetector/cameratraps
export PYTHONPATH=$PYTHONPATH:$PWD/megadetector/ai4eutils
```

4. Create the conda virtual environment from microsoft's .yml file. See [here](https://github.com/microsoft/CameraTraps#installation) for more.
```
conda env create --file cameratraps/environment-detector.yml
```

5. Activate the conda environment. See [here](https://github.com/microsoft/CameraTraps#installation) for more.
```
conda activate cameratraps-detector
```

6. Install additional packages into the Python environment.
```
conda install pandas pytorch torchvision
```

7. Run Microsoft's run_tf_detector_batch.py script to use the Megadetector on the raw Wellington Camera Traps images and generate the detections output file detections.json. See [2. run_tf_detector_batch.py](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md#2-run_tf_detector_batchpy) for more.
```
python ./megadetector/cameratraps/detection/run_tf_detector_batch.py ./megadetector/megadetector_v3.pb ./data/raw ./megadetector/detections.json
```

More detailed instructions and trouble shooting for using the megadetector can be found [here](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md) and [here](https://github.com/microsoft/CameraTraps#installation).

### 4. Prepare Classification Training and Testing Dataset
Generate and process the classification dataset images, sort them into training and testing datasets, and clean the metadata.

1. Generate the classification dataset images by running the crop_detections.py script to crop the animals from the raw images using the detections generated in step 3. Run the Microsoft's Megadetector. Images labeled as empty according to the Wellington Camera Traps metadata file are cropped using the boundary box coordinates of the previous detection.
```
python crop_detections.py ./megadetector/detections.json ./data/raw/wellington_camera_traps.csv ./data/raw ./data/detections
```

2. Sort the cropped images into ten training and seven testing datasets using the prepare_classification_datasets.py script
```
python prepare_classification_datasets.py --detections_csv=./data/detections/wellington_camera_traps_detections.csv --output_csv=training_testing_datasets.csv
```
This script also:
- Combines images labeled as rat, Norway rat, and ship rat into a single "rat" label.
- Combines images labeled as hare and rabbit into a single "hare" label.
- Changes images labeled as nothinghere to empty.
- Generates two labels for each image:
    - An individual label specifying the species or group of species.
    - A grouped label where birds are labeled as "native," empty frames are labeled as "empty," and all others (unclassifiable, cat, deer, dog, hare, hedgehog, mouse, mustelid, pig, possum, and rat) are labeled as "invasive."

### 5. Training
Train the ten convolutional neural network models using each of the ten training datasets with the training.py script.

Use the following command to start training fresh.
```
python training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections
```

Use the following command to resume training after stopping from saved checkpoints.
```
python training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections --resume_training
```

Each model uses the ResNet-18 architecture[[3]](#references) pretrained on the imagenet database[[4]](#references) with a rectified linear activation function (ReLU), Stochastic Gradient Descent with momentum for backpropagation[[5]](#references), and learning rates and weight decays matching those of Norouzzadeh et al. (2018)[[6]](#references) and Tabak et al. (2018)[[7]](#references).

Standard data augmentation techniques (random horizontal flipping, cropping, and color jitter) are applied[[8]](#references).


### 6. Testing and Evaluating
Test and evaluate the ten models using the seven testing datasets in 13 combinations using the training.py script with the --evaluate flag.
```
python training.py --training_testing_csv ./training/training_testing_datasets.csv --dataset_dir ./data/detections --evaluate
```
This script generates predictions for each train/test combination and stores them in a .csv file in the training/predictions directory and a single file called model_performance_results.csv containing the following performance metrics across each train/test combination, and by site and class/label within each train/test combination:
- Top-1 accuracy: # of times the top prediction is correct / # of images tested
- Top-5 accuracy: # of times the correct class is in the top 5 predictions / # of images tested (models with five or fewer output classes will always have top-5 accuracies of 100%)
- False alarm rate: # of empty or native images labeled as invasive / # of images tested
- Missed invasive rate: # of invasive images labeled as native or empty / # of invasive images tested

### 7. Figures
Generate the paper figures with the generate_figures.py script.

1. Install additional packages into the Python environment.
```
conda install bokeh seaborn
```

2. Generate figures.
```
python generate_figures.py
```


## Acknowledgement
I would like to thank my Academic Advisors at San José State University:
Lynne Trulio, Ph.D.    -  Department of Environmental Studies
Dustin Mulvaney, Ph.D. -  Department of Environmental Studies
Philip Heller, Ph.D.   -  Department of Computer Science

## References

1. Anton, V., Hartley, S., Geldenhuis, A., & Wittmer, H. U. (2018).  Monitoring the mammalian fauna of urban areas using remote cameras and citizen science. Journal of Urban Ecology,4(1). https://doi.org/10.1093/jue/juy002

2. Beery, S., Morris, D., & Yang, S. (2019).  Efficient pipeline for camera trap image review.  arXiv. https://arxiv.org/abs/1907.06772

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016).  Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778. https://doi.org/10.1109/CVPR.2016.90

4. Deng, J., Dong, W., Socher, R., Li, L., Li, K., & Fei-Fei, L.(2009).  ImageNet: A large-scale hierarchical image database. 2009 IEEEConference on Computer Vision and Pattern Recognition, 248–255. https://doi.org/10.1109/CVPR.2009.5206848

5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning.  The MIT Press.

6. Norouzzadeh, M. S., Nguyen, A., Kosmala, M., Swanson, A., Palmer, M. S., Packer, C., & Clune, J. (2018). Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning. Proceedings of the National Academy of Sciences - PNAS,115(25), Article E5716–E5725. https://doi.org/10.1073/pnas.1719367115

7. Tabak, M. A., Norouzzadeh, M. S., Wolfson, D. W., Sweeney, S. J., Vercauteren, K. C., Snow, N. P., Halseth, J. M., Di Salvo, P. A., Lewis, J. S., White, M. D., Teton, B., Beasley, J. C., Schlichting, P. E., Boughton, R. K.,Wight, B., Newkirk, E. S., Ivan, J. S., Odell, E. A., Brook, R. K., ... Miller, R. S. (2019). Machine learning to classify animal species in camera trap images: Applications in ecology. Methods in Ecology and Evolution,10(4), 585–590. https://doi.org/10.1111/2041-210X.13120

8. Shorten, C., & Khoshgoftaar, T. M. (2019).  A survey on image data augmentation for deep learning.  Journal of Big Data, 6(1), 1–48.  https://doi.org/10.1186/s40537-019-0197-0


