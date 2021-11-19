# README
## Introduction
The source code was trying to reproduce the paper - "Micro-expression Recognition Based on Facial Graph Representation Learning and Facial Action Unit Fusion". [[paper]](https://openaccess.thecvf.com/content/CVPR2021W/AUVi/papers/Lei_Micro-Expression_Recognition_Based_on_Facial_Graph_Representation_Learning_and_Facial_CVPRW_2021_paper.pdf) [[official code]](https://github.com/raying777/FGRMER)

## Installation

### Requirements
```command
# Install requirement
$ pip install -r requirements.txt

# Download landmarks weight for DLIB
$ mkdir -p dataloader/weight
$ wget https://github.com/davisking/dlib-models/raw/master/mmod_human_face_detector.dat.bz2 -P dataloader/weight
$ bzip2 -d dataloader/weight/mmod_human_face_detector.dat.bz2
$ wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 -P dataloader/weight
$ bzip2 -d dataloader/weight/shape_predictor_68_face_landmarks.dat.bz2
```

### MagNet
The structure of MagNet was adapted from [here](https://github.com/ZhengPeng7/motion_magnification_learning-based). Please download the pretrained weight from their release and place in `dataloader/weight/`.

### DLIB with GPU (not necessary)
```command
# Remove the cpu version first
$ pip uninstall dlib
# Install cudnn and its toolkit
$ conda install cudnn cudatoolkit
# Build from source
$ git clone https://github.com/davisking/dlib.git
$ cd dlib
$ mkdir build & cd build
$ cmake .. \
    -DDLIB_USE_CUDA=1 \
    -DUSE_AVX_INSTRUCTIONS=1 \
    -DCMAKE_PREFIX_PATH=<path to  conda env>\
    -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6
$ cmake --build .
$ cd ..
$ python setup.py install \
    --set USE_AVX_INSTRUCTIONS=1 \
    --set DLIB_USE_CUDA=1 \
    --set CMAKE_PREFIX_PATH=<path to  conda env>  \
    --set CMAKE_C_COMPILER=gcc-6 \
    --set CMAKE_CXX_COMPILER=g++-
```

## Dataset
* [CASME II](http://fu.psych.ac.cn/CASME/casme2-en.php)
* [SAMM](https://personalpages.manchester.ac.uk/staff/adrian.davison/SAMM.html)

## Training
```
usage: train.py [-h] --csv_path CSV_PATH --image_root IMAGE_ROOT --npz_file
                NPZ_FILE --catego CATEGO [--num_classes NUM_CLASSES]
                [--batch_size BATCH_SIZE]
                [--weight_save_path WEIGHT_SAVE_PATH] [--epochs EPOCHS]
                [--learning_rate LEARNING_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --csv_path CSV_PATH   Path for the csv file for training data
  --image_root IMAGE_ROOT
                        Root for the training images
  --npz_file NPZ_FILE   Files root for npz
  --catego CATEGO       SAMM or CASME dataset
  --num_classes NUM_CLASSES
                        Classes to be trained
  --batch_size BATCH_SIZE
                        Training batch size
  --weight_save_path WEIGHT_SAVE_PATH
                        Path for the saving weight
  --epochs EPOCHS       Epochs for training the model
  --learning_rate LEARNING_RATE
                        Learning rate for training the model
```

## Citation
```bibtex
@InProceedings{Lei_2021_CVPR,
    author    = {Lei, Ling and Chen, Tong and Li, Shigang and Li, Jianfeng},
    title     = {Micro-Expression Recognition Based on Facial Graph Representation Learning and Facial Action Unit Fusion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {1571-1580}
}
```
