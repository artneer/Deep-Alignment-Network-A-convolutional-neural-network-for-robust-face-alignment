Deep Alignment Network: A convolutional neural network for robust face alignment
===

This is a **Tensorflow** implementations of paper *"Deep Alignment Network: A convolutional neural network for robust face alignment"*.
You can see **Original implementation** [here](https://github.com/MarekKowalski/DeepAlignmentNetwork).

-----------------

## System

* Intel Core i5-6600K CPU @ 3.5GHz
* 8.00GB RAM
* NVIDIA Gefore GTX 960
* Windows 10 64bit

Getting started
-------  
First of all you need to install CUDA Toolkit and cuDNN. 
I recommend Anaconda 3, because it has all the neccessary libraries except for tensorflow and opencv.

* Anaconda 3 (Python 3.5.6) [download](https://www.anaconda.com/products/individual)
* CUDA v9.0.176 for Windows 10 (Sept. 2017) [download](https://developer.nvidia.com/cuda-toolkit-archive)
* cuDNN v7.0.5 for Windows 10 (Dec. 5, 2017) [download](https://developer.nvidia.com/rdp/cudnn-archive)
* Tensorflow 1.9.0
* OpenCV 4.2.0

Tensorflow and OpenCV can be installed with the following commands:
```shell
pip install --upgrade opencv-python
pip install --upgrade tensorflow-gpu==1.9.0
```

How to prepare dataset
---
* Download the 300W, LFPW, HELEN, AFW and IBUG datasets from [https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) and extract them /DAN_V2/db/ into seperate directories: 300W, lfpw, helen, afw and ibug. Run the under script, it may take a while.
```shell
python split_trainset.py
python split_testset.py
```
* Write mirror file. There is a 68 landmark mirror file. [download](https://pan.baidu.com/s/1Ln_i00DRulDlgHJ8CmIqAQ)
* Preprocess.
```shell
python preprocessing.py --input_dir=./data/train --output_dir=./prep/train --istrain=True --repeat=10 --img_size=112 --mirror_file=./Mirror68.txt
python preprocessing.py --input_dir=./data/valid --output_dir=./prep/valid --istrain=False --img_size=112
python preprocessing.py --input_dir=./data/test/common_set --output_dir=./prep/test/common_set --istrain=False --img_size=112
python preprocessing.py --input_dir=./data/test/challenge_set --output_dir=./prep/test/challenge_set --istrain=False --img_size=112
python preprocessing.py --input_dir=./data/test/300w_private_set --output_dir=./prep/test/300w_private_set --istrain=False --img_size=112
```

How to train 300w model
---
```shell
python DAN_V2.py -ds 1 --data_dir=./prep/train --data_dir_test=./prep/valid -nlm 68 -te=15 -epe=1 -mode train
python DAN_V2.py -ds 2 --data_dir=./prep/train --data_dir_test=./prep/valid -nlm 68 -te=45 -epe=1 -mode train
```

How to evaluate acc.
---
First of all, execute following script. You can get result points on './prep/predict'. Before executing next script, copy the files in the folders ('./prep/predict/common_set', './prep/predict/challenge_set', './prep/predict/300w_private_set') seperately.
```shell
python DAN_V2.py -ds 2 --data_dir=./prep/test/common_set --data_dir_test=None -nlm 68 -mode predict
python DAN_V2.py -ds 2 --data_dir=./prep/test/challenge_set --data_dir_test=None -nlm 68 -mode predict
python DAN_V2.py -ds 2 --data_dir=./prep/test/300w_private_set --data_dir_test=None -nlm 68 -mode predict
```
For calculating the errors, execute the command:
```shell
python dan_predict.py
```

Results on 300W
---
* Err : `1.54 %` on 300W common subset (bounding box diagonal normalization).
* Err : `2.49 %` on 300W challenge subset (bounding box diagonal normalization).
* Err : `1.73 %` on 300W full set (bounding box diagonal normalization).

Pre-trained Model
---
TODO:You can download pre-trained model [here](). This model trained on 300W dataset.
