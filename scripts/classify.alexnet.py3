#!/usr/bin/env python3
"""
Author: jpc5731@cse.psu.edu (Jinhang Choi), 8/4/2016

Original version is Berkeley Vision LC, $CAFFE/python/classify.py
classify.py is an out-of-the-box image classifer callable from the command line.
By default it configures and runs the fine-tuned Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import lmdb
import argparse
import time

import caffe

sys.stdout = sys.stderr

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--test_db",
        help="Input image database."
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir, "models/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir, "models/caffenet_train_iter_30000.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        default=-1,
        dest='gpu_id',
        help="Switch for gpu computation.",
        type=int
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir, 'imagenet/imagenet_mean.binaryproto'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--channel_swap",
        default="0,1,2",
        help="Order to permute input channels. " +
             "If you created lmdb by using Caffe, the channel format would already be BGR."
    )
    parser.add_argument(
        "--synset_words",
        default='synset_words.txt',
        help="Object id and name list."
    )
    parser.add_argument(
        "--top_k",
        default='3',
        help="Number of candidate."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]
    mean, channel_swap = None, None

    if args.mean_file:
        print("MeanFile: %s\n" % args.mean_file)
        if args.mean_file.endswith('binaryproto'):
            blob = caffe.proto.caffe_pb2.BlobProto()
            data = open(args.mean_file, 'rb').read()
            blob.ParseFromString(data)
            mean = caffe.io.blobproto_to_array(blob)[0]
        elif args.mean_file.endswith('npy'):
            mean = np.load(args.mean_file)

    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    # Set How to use Caffe: GPU or CPU?
    if args.gpu_id > -1:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    top_k = int(args.top_k)
    # Index of Object ID
    categories = []
    index = dict()
    for line in np.loadtxt(args.synset_words, str, delimiter="\t"):
        categories.append(line[0])
        index.update({line[0]:line[1]})

    # Make classifier.
    print("load tranied model")

    # Since classifier tries to oversampling, it is important to directly invoke Net
    # so that prediction can avoid image manipulation
    net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)

    # Choose the position of cropped center
    crop_dims = np.array(net.blobs['data'].data.shape[2:])
    center = np.array(image_dims) / 2.0
    crop = np.tile(center, (1, 2))[0] + np.concatenate([-crop_dims / 2.0, crop_dims/2.0])

    # Set up transformer configuration
    # input  : Height x Width x Channel
    # output : Channel x Height x Width
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data',(2,0,1))
    if mean is not None:
        transformer.set_mean('data', mean[:, int(crop[0]):int(crop[2]), int(crop[1]):int(crop[3])])
    transformer.set_channel_swap('data', channel_swap)
    # Image data in LMDB already has raw scale of 255
    # Hence, you do not need to set up raw_scale property in transformer

    # batch size
    repeat = net.blobs['data'].data.shape[0]

    # load LMDB input
    prediction_time = 0
    g_start = time.time()
    db = lmdb.open(args.test_db, readonly=True)
    datum = caffe.proto.caffe_pb2.Datum()

    with db.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum.ParseFromString(value)
            label = index[categories[datum.label]]
            idx = int(key.decode('ascii').split('_')[0]);
            data = caffe.io.datum_to_array(datum)
            data = np.asarray([data])

            # Accumulate the set of input images before calling classifier
            if((idx % repeat) == 0):
                inputs = data
            else:
                inputs = np.concatenate((inputs,data), axis=0)

            if((idx % repeat) < repeat-1):
              continue

            # take center crop based on input data layer
            # convert dtype to floating-point for considering mean subtraction
            inputs = inputs[:, :, int(crop[0]):int(crop[2]), int(crop[1]):int(crop[3])].astype(np.float32)

            # change inputs according to transformer definition
            # lmdb data format  : Channel x Height x Width
            # transformer input : Height x Width x Channel
            for idx, img in enumerate(inputs):
                inputs[idx] = transformer.preprocess('data', img.transpose(1,2,0))

            if((idx % (repeat*10)) == (repeat*10-1)):
                print("[ %.3f ]" % prediction_time, "Classifying %s item." % key)

            # Classify.
            l_start = time.time()
            out = net.forward_all(data=inputs)
            predictions = out['prob']
            time_step = time.time() - l_start
            prediction_time += time_step

            # Sorting inference results of the last-inserted input among batch images
            prediction = zip(predictions[repeat-1].flatten().tolist(), categories)
            #prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
            prediction = sorted(prediction, key=lambda x: x[0], reverse=True)
            print(" ground_truth: %s" % label)
            for rank, (score, name) in enumerate(list(prediction)[:top_k], start=1):
                print('  #%d | %s | %4.1f%%' % (rank, index[name], score * 100))


    print(" Globally Done in %.3f s (prediction: %.3f s)." % ((time.time() - g_start),  prediction_time))

if __name__ == '__main__':
    main(sys.argv)
