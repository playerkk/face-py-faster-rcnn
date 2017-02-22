# Face Detection with the Faster R-CNN

This repository contains source files of face detection using the Faster R-CNN. It is developed based on the awesome [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) repository. 

For technical details, please refer to the [technial report](https://arxiv.org/pdf/1606.03473v1.pdf) here. Faster R-CNN was initially described in the [NIPS 2015 paper](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf). The approximate joint end-to-end training was described in the [PAMI paper](https://arxiv.org/pdf/1506.01497v3.pdf). Please consider cite these papers if you find this repository useful for your research.

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download the [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) (from Ross Girshick) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

1. For training smaller networks (ZF, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training Fast R-CNN with VGG16, you'll need a K40 (~11G of memory)
3. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the face Faster R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive git@github.com:playerkk/face-py-faster-rcnn.git
  ```

2. We'll call the directory that you cloned Faster R-CNN into `FRCN_ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*

   **Note 1:** If you didn't clone Faster R-CNN with the `--recursive` flag, then you'll need to manually clone the `caffe-fast-rcnn` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `caffe-fast-rcnn` submodule needs to be on the `faster-rcnn` branch (or equivalent detached state). This will happen automatically *if you followed step 1 instructions*.

3. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

5. Download pre-computed Faster R-CNN detectors
    ```Shell
    cd $FRCN_ROOT
    ./data/scripts/fetch_faster_rcnn_models.sh
    ```

    This will populate the `$FRCN_ROOT/data` folder with `faster_rcnn_models`. See `data/README.md` for details.
    These models were trained on VOC 2007 trainval.

### Prepare training data
1. Download the WIDER face dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). Extract all files into one directory named WIDER.

2. It should have this basic structure

  ```Shell
    $WIDER/                                 # data directory
    $WIDER/WIDER_train/                     # training set
    $WIDER/WIDER_val/                       # validation set
    # ... and several other directories ...
    ```
    
3. Download the [file](http://jianghz.me/files/wider_face_train_annot.txt) and put it under the WIDER directory. It contains annotations of the training images, which follows the annotation format of [FDDB](http://vis-www.cs.umass.edu/fddb/README.txt). Faces that are less than 10 pixels are discarded (considered as background), which slightly improves the performance.

4. Create symlinks for the WIDER dataset

  ```Shell
    cd $FRCN_ROOT/data
    ln -s $WIDER WIDER
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. Follow the next sections to download pre-trained ImageNet models

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage

To train a Faster R-CNN face detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] wider [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

To test the trained model, run

```
python ./tools/run_face_detection_on_fddb.py --gpu=0
```

### Pre-trained face detection model
A pre-trained face detection model trained on the WIDER training set is available [here](http://supermoe.cs.umass.edu/~hzjiang/data/vgg16_faster_rcnn_iter_80000.caffemodel).

###Acknowledgment 
This research is based upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA) under contract number 2014-14071600010. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purpose notwithstanding any copyright annotation thereon.
