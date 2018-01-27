#!/usr/bin/env python3
"""
Author: jpc5731@cse.psu.edu (Jinhang Choi), 1/27/2018

Original version is Berkeley Vision LC, $CAFFE/python/classify.py
classify.py is an out-of-the-box image classifer callable from the command line.
By default it configures and runs the fine-tuned Caffe reference ImageNet model.
"""
import os
import sys
import argparse
import time
import math
import csv
import lmdb
import numpy as np
import cv2

import caffe

sys.stdout = sys.stderr

SIGMA = (2.0, 8.0)  # gaussian range (smaller -> circle, larger -> line)
LAMBDA = (8.0,)     # thickness
GAMMA = 1           # x' versus y'
PSI = 0             # center
GABORS = []

def gabor_convolution(im_rgb, cell_size, padding, stride):
    '''Perform cell-wise gabor filter-based convolution'''
    height = im_rgb.shape[0]
    width = im_rgb.shape[1]
    channels = im_rgb.shape[2]

    # add padding to the image
    im_rgb_padded = np.pad(im_rgb, ((padding, padding), (padding, padding), (0, 0)), \
                            mode='constant', constant_values=0)
    out_height = (height + 2*padding - cell_size[0])//stride + 1
    out_width = (width + 2*padding - cell_size[1])//stride + 1

    # im2col indexing
    row_offset = np.repeat(np.arange(cell_size[0]), cell_size[1])
    row_base = stride * np.repeat(np.arange(out_height), out_width) + padding
    row_idx = row_offset.reshape(-1, 1) + row_base.reshape(1, -1)

    col_offset = np.tile(np.arange(cell_size[1]), cell_size[0])
    col_base = stride * np.tile(np.arange(out_width), out_height) + padding
    col_idx = col_offset.reshape(-1, 1) + col_base.reshape(1, -1)

    # serialize row, col, and channel of image pixels and filters
    cols = im_rgb_padded[row_idx, col_idx, :].astype('f')
    cols = cols.reshape(cell_size[0]*cell_size[1], -1)
    gabor = np.array(GABORS, dtype='f').reshape(len(GABORS), -1)

    # mat-mat multiplication
    # start to convolve gabor filter according to theta = {0, pi/max_bins, ..., pi*(1-1/max_bins)}
    out = np.matmul(gabor, cols)

    # reshape
    im_features = out.reshape(len(GABORS), out_height, out_width, channels)
    # change row x col x ch -> ch x row x col, and migrate features from 4D to 3D
    im_features = im_features.transpose(3, 0, 1, 2).reshape(-1, out_height, out_width)

    return im_features

def prepare_args():
    '''
    A list of arguments can be used in this python script
    '''
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
        #default=os.path.join(pycaffe_dir, 'imagenet/imagenet_mean.binaryproto'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
        "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--gabor",
        #default="9,1,8,3",
        help="Activate Gabor Feature Extractor. " +
        "Kernel size, Padding size, Stride size, Angular bin count."
    )
    parser.add_argument(
        "--verbose",
        help="Analyze Front-end Feature Extraction."
    )
    parser.add_argument(
        "--channel_swap",
        #default="0,1,2",
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

    return parser.parse_args()

def main(argv):
    '''
    Run CNN top-k classification by repeating the retrieval of an input batch from test LMDB
    '''
    args = prepare_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]
    gabor_param, mean, channel_swap = None, None, None

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

    if args.gabor:
        gabor_param = [int(s) for s in args.gabor.split(',')]
        cell_size = (gabor_param[0], gabor_param[0])
        padding = gabor_param[1]
        stride = gabor_param[2]
        # Since there is no difference between clockwise/anti-clockwise degree,
        # I am going to just check 0 ~ 180 degrees
        max_bins = gabor_param[3]
        unit_degree = math.pi / max_bins

        # gabor_complex   = exp(-(x'^2+gamma^2*y'^2)/(2*sigma^2))*exp(i(2*pi*x'/lambda + psi))
        # gabor_real      = exp(-(x'^2+gamma^2*y'^2)/(2*sigma^2))*cos(2*pi*x'/lambda + psi)
        # gabor_imaginary = exp(-(x'^2+gamma^2*y'^2)/(2*sigma^2))*sin(2*pi*x'/lambda + psi)
        # such that
        #   x' = x*cos(theta) + y*sin(theta)
        #   y' = -x*sin(theta) + y*cos(theta)
        for sigma in SIGMA:
            for thick in LAMBDA:
                for bins in range(max_bins):
                    # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
                    #   ksize - size of gabor filter (n, n)
                    #   sigma - standard deviation of the gaussian function
                    #   theta - orientation of the normal to the parallel stripes
                    #   lambda - wavelength of the sunusoidal factor
                    #   gamma - spatial aspect ratio
                    #   psi - phase offset
                    #   ktype - type and range of values that each pixel
                    #           in the gabor kernel can hold
                    GABORS.append(cv2.getGaborKernel(cell_size, sigma, bins * unit_degree,\
                                                    thick, GAMMA, PSI, ktype=cv2.CV_32F))

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
    crop = np.tile(center, (1, 2))[0] + np.concatenate([-crop_dims/2.0, crop_dims/2.0])

    # Set up transformer configuration
    # input  : Height x Width x Channel
    # output : Channel x Height x Width
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    if mean is not None:
        transformer.set_mean('data', mean[:, int(crop[0]):int(crop[2]), int(crop[1]):int(crop[3])])
    if channel_swap is not None:
        transformer.set_channel_swap('data', channel_swap)
    # Image data in LMDB already has raw scale of 255
    # Hence, you do not need to set up raw_scale property in transformer

    # batch size
    repeat = net.blobs['data'].data.shape[0]

    # open csv file (for output)
    csvfile_fp, csvfile = None, None
    if args.output:
        csvfile_fp = open(args.output+"_top"+args.top_k+".csv", 'w')
        csvfile = csv.writer(csvfile_fp,
                             delimiter=',',
                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_line = ['item_no', 'ground_truth']
        for i in range(top_k):
            csv_line.extend(['label', 'score'])
        csvfile.writerow(csv_line)

    # load LMDB input
    prediction_time = 0
    g_start = time.time()
    testdb = lmdb.open(args.test_db, readonly=True)
    datum = caffe.proto.caffe_pb2.Datum()

    with testdb.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum.ParseFromString(value)
            label = index[categories[datum.label]]
            idx = int(key.decode('ascii').split('_')[0])
            data = caffe.io.datum_to_array(datum)
            data = np.asarray([data])

            # Accumulate the set of input images before calling classifier
            if (idx % repeat) == 0:
                inputs = data
            else:
                inputs = np.concatenate((inputs, data), axis=0)

            if (idx % repeat) < repeat-1:
                continue

            csv_line = [key.decode('ascii'), categories[datum.label]]

            if gabor_param is not None:
                prep_time = 0
                for i, img in enumerate(inputs):
                    elem = img.transpose(1, 2, 0).astype('float')
                    l_start = time.time()
                    ifeat = gabor_convolution(elem, cell_size, padding, stride)

                    # exclude memory reconstruction from prep_time with respect to the batch size
                    prep_time = prep_time + time.time() - l_start
                    # stack 1 x ch x row x col
                    ifeat = np.reshape(ifeat, \
                                (1, ifeat.shape[0], ifeat.shape[1], ifeat.shape[2]))
                    if i == 0:
                        new_inputs = ifeat
                    else:
                        new_inputs = np.concatenate((new_inputs, ifeat), axis=0)
                inputs = new_inputs

                l_start = time.time()
                # take center crop based on input data layer
                inputs = inputs[:, :, \
                                int(crop[0]):int(crop[2]), int(crop[1]):int(crop[3])].astype('f')
                prep_time = prep_time + time.time() - l_start
            else:
                l_start = time.time()

                # take center crop based on input data layer
                # convert dtype to floating-point for considering mean subtraction
                inputs = inputs[:, :, \
                                int(crop[0]):int(crop[2]), int(crop[1]):int(crop[3])].astype('f')
                prep_time = time.time() - l_start

                if mean is not None:
                    # change inputs according to transformer definition
                    # lmdb data format  : Channel x Height x Width
                    # transformer input : Height x Width x Channel
                    for i, img in enumerate(inputs):
                        elem = img.transpose(1, 2, 0)
                        l_start = time.time()
                        inputs[i] = transformer.preprocess('data', elem)
                        prep_time = prep_time + time.time() - l_start

            if(idx % (repeat*10)) == (repeat*10-1):
                print("[ %.3f ]" % prediction_time, "Classifying %s item." % key)

            # Classify.

            if args.verbose is not None:
                layer_sep = [str(s) for s in args.verbose.split(',')]
                #print (layer_sep)
                l_start = time.time()
                net.forward(start='data', end=layer_sep[0], data=inputs)
                #print(net.blobs[layer_sep[0]].data[...])
                front_time = time.time()
                out = net.forward(start=layer_sep[1])
                predictions = out['prob']
                time_step = time.time() - front_time
                front_time = front_time - l_start
                prediction_time += front_time
            else:
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
                csv_line.extend([name, str(score*100)])
            if args.verbose is not None:
                print("  * Locally Done in %.2f ms(prep) + %.2f/%.2f ms(infer)." % \
                            (prep_time * 1000, front_time * 1000, time_step * 1000))
            else:
                print("  * Locally Done in %.2f ms(prep) + %.2f ms(infer)." % \
                            (prep_time * 1000, time_step * 1000))

            if csvfile is not None:
                csvfile.writerow(csv_line)
                csvfile_fp.flush()

    print(" Globally Done in %.3f s (prediction: %.3f s)." % \
                    ((time.time() - g_start), prediction_time))

if __name__ == '__main__':
    main(sys.argv)
