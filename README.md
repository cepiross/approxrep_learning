# Approximated front-end representation-based Learning

This repository presents source code necessary to reproduce training/inference addressed in the following paper:
```
@unpublished{Choi2018ICCD_ApproxRepLearning,
  title={{Heuristic Approximation of Early-Stage CNN Data Representation for Vision Intelligence Systems}},
  author={Jinhang Choi and Jack Sampson and Vijaykrishnan Narayanan},
  note={IEEE International Conference on Computer Design (ICCD)},
  year={2018}
}
```

This repository is composed of the following items:
  - notebook/
    * Experimental reports
  - android-cmake/
    * Android NDK compile setup
  - caffe-android-demo/
    * Android App for testing image classification using Caffe Library
  - caffe-android-lib/
    * Caffe JNI Library for Android Marshmallow on armv7a uarch
  - data/
    * ilsvrc2012 synset id for training dataset, validation dataset, and inference tests
    * ilsvrc2012 mean substraction for raw images
  - models/
    * [BVLC CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) (AlexNet) - Reference / HOG+modified version / Gabor+modified version
    * [SqueezeNet](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.0) - Reference / HOG+modified version / Gabor+modified version
    * [BVLC GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) - Reference / HOG+modified version / Gabor+modified version
    * It is also possible to train CNNs by using prototxt descriptors
  - scripts/
    * classify.alexnet.gabor.py3 - inference test for gabor filters by reading LMDB (Gabor feature LMDB)
    * classify.alexnet.hog.py3 - inference test for hog by reading LMDB (either original raw image LMDB or HOG feature LMDB)
    * cvtlmdb_img2gabor.py  - LMDB converter from raw images to Gabor features
    * cvtlmdb_img2hog.py    - LMDB converter from raw images to HOG features 
    * cvtlmdb_img2resize.py - LMDB converter by resizing raw images
