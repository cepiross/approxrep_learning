#!/usr/bin/env python3
"""
Author: jpc5731@cse.psu.edu (Jinhang Choi), 8/4/2016

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
import concurrent.futures
import lmdb
import cv2
import numpy as np

import caffe

sys.stdout = sys.stderr

# I am going to try anti-aliasing according to factor x supersampling
NUM_BINS = 6
ALIASING_FACTOR = 3
MAX_BINS = NUM_BINS * ALIASING_FACTOR
UNIT_DEGREE = 2 * math.pi / MAX_BINS
TRIGON_PAIRS = np.zeros((2, MAX_BINS), dtype='f')
for tri_idx in range(MAX_BINS):
    TRIGON_PAIRS[0][tri_idx] = math.cos(tri_idx * UNIT_DEGREE)
    TRIGON_PAIRS[1][tri_idx] = math.sin(tri_idx * UNIT_DEGREE)

def compute_candidate(im_candidate, im_ix, im_iy, bin_idx):
    '''
    Compute intensity of gradient candidates with respect to Ix, Iy, and angular bin
    '''
    im_candidate[bin_idx, ...] = np.add(np.multiply(im_ix[...], TRIGON_PAIRS[0][bin_idx]), \
                                       np.multiply(im_iy[...], TRIGON_PAIRS[1][bin_idx]))

def forward_interpolate_gradient(im_gradient, bin_idx):
    '''
    Propagate gradients from intermediate angular bins to the next target angular bin.
    prev_bin -- (intermediate: prev_bin+offset) -- next_bin
    '''
    if bin_idx % ALIASING_FACTOR == 0 and bin_idx < MAX_BINS:
        next_bin = (bin_idx + ALIASING_FACTOR) % MAX_BINS
        for offset in range(1, ALIASING_FACTOR):
            im_gradient[next_bin, ...] = np.add(im_gradient[next_bin, ...], \
                                               np.multiply(im_gradient[bin_idx+offset, ...], \
                                               float(offset) / ALIASING_FACTOR))

def backward_interpolate_gradient(im_gradient, bin_idx):
    '''
    Propagate gradients from intermediate angular bins to the previous target angular bin.
    prev_bin -- (intermediate: prev_bin+offset) -- next_bin
    '''
    if bin_idx % ALIASING_FACTOR == 0 and bin_idx > 0:
        prev_bin = bin_idx - ALIASING_FACTOR
        for offset in range(1, ALIASING_FACTOR):
            im_gradient[prev_bin, ...] = np.add(im_gradient[prev_bin, ...], \
                                               np.multiply(im_gradient[prev_bin+offset, ...], \
                                               float(ALIASING_FACTOR - offset) / ALIASING_FACTOR))

def hog_histogram_parallel(im_rgb, param):
    '''
    Make cell-wise histogram of oriented gradients
    '''
    height = im_rgb.shape[0]
    width = im_rgb.shape[1]
    cell_size = param[0]
    stride = param[1]
    # start to compute element-wise gradient, magnitude, and max orientation
    # according to Ix * cos(theta) + Iy * sin(theta)
    # where theta = {0, 2*pi/max_bins, ..., 2*pi*(1-1/max_bins)}
    im_ix = cv2.filter2D(im_rgb, -1, np.array([[-1, 0, 1]])).transpose(2, 0, 1)
    im_iy = cv2.filter2D(im_rgb, -1, np.array([[-1], [0], [1]])).transpose(2, 0, 1)
    im_candidate = np.zeros((MAX_BINS, im_rgb.shape[2], im_rgb.shape[0], im_rgb.shape[1]), \
                            dtype='f')
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_BINS) as executor:
        futures = [executor.submit(compute_candidate, im_candidate, im_ix, im_iy, i) \
                    for i in range(MAX_BINS)]
        concurrent.futures.wait(futures)

    im_candidate = im_candidate.astype('int16')
    im_orientation = np.argmax(im_candidate, axis=0).astype('uint8')
    # spread magnitude with respect to orientation theta
    # such that argmax Ix*cos+Iy*sin = Msin(theta-alpha) where theta = alpha
    im_gradient = np.zeros_like(im_candidate)
    idx = np.indices(im_orientation.shape)
    im_gradient[im_orientation, idx[0], idx[1], idx[2]] = \
                im_candidate[im_orientation, idx[0], idx[1], idx[2]]

    if ALIASING_FACTOR > 1:
        # bilinear interploation of magnitude to mitigate aliasing
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_BINS) as executor:
            futures = [executor.submit(forward_interpolate_gradient, im_gradient, i) \
                            for i in range(0, MAX_BINS, ALIASING_FACTOR)]
            concurrent.futures.wait(futures)
            futures = [executor.submit(backward_interpolate_gradient, im_gradient, i) \
                            for i in range(MAX_BINS, 0, -ALIASING_FACTOR)]
            concurrent.futures.wait(futures)

        # downsample MAX_BINS to NUM_BINS
        im_gradient = im_gradient[::ALIASING_FACTOR, ...]

    # record histogram of oriented gradients (magnitude)
    im_histogram = np.zeros((NUM_BINS, im_rgb.shape[2], \
                                (im_rgb.shape[0]-cell_size)//stride+1, \
                                (im_rgb.shape[1]-cell_size)//stride+1))

    for row in range(0, height-cell_size+1, stride):
        for col in range(0, width-cell_size+1, stride):
            im_histogram[..., row//stride, col//stride] = \
                        np.sum(im_gradient[..., row:row+cell_size, col:col+cell_size], axis=(2, 3))

    # since the order of dimensions in previous implementation: (HEIGHT, WIDTH, CHANNEL, BINS),
    # optimized algorithm should also follow the order of dimensions: (CHANNEL, BINS, HEIGHT, WIDTH)
    return im_histogram.transpose(1, 0, 2, 3)

def hog_histogram(im_rgb, param):
    '''Make cell-wise histogram of oriented gradients'''
    height = im_rgb.shape[0]
    width = im_rgb.shape[1]
    cell_size = param[0]
    stride = param[1]

    # start to compute element-wise gradient, magnitude, and max orientation
    # according to Ix * cos(theta) + Iy * sin(theta)
    # where theta = {0, 2*pi/max_bins, ..., 2*pi*(1-1/max_bins)}
    im_ix = cv2.filter2D(im_rgb, -1, np.array([[-1, 0, 1]]))
    im_iy = cv2.filter2D(im_rgb, -1, np.array([[-1], [0], [1]]))

    im_candidate = np.zeros((im_rgb.shape[0], im_rgb.shape[1], im_rgb.shape[2], MAX_BINS))
    for i in range(MAX_BINS):
        im_candidate[:, :, :, i] = np.add(np.multiply(im_ix[:, :, :], TRIGON_PAIRS[0][i]), \
                                         np.multiply(im_iy[:, :, :], TRIGON_PAIRS[1][i]))
    im_orientation = np.argmax(im_candidate, axis=3)
    im_gradmax = np.max(im_candidate, axis=3)

    # spread magnitude with respect to orientation theta
    # such that argmax Ix*cos+Iy*sin = Msin(theta-alpha) where theta = alpha
    im_gradient = np.zeros_like(im_candidate)
    for i in range(MAX_BINS):
        loc = np.where(im_orientation == i)
        im_gradient[loc[0], loc[1], loc[2], i] = im_gradmax[loc[0], loc[1], loc[2]]

    if ALIASING_FACTOR > 1:
        # bilinear interploation of magnitude to mitigate aliasing
        for i in range(MAX_BINS):
            offset = i % ALIASING_FACTOR
            if offset != 0:
                prev_bin = i - offset
                next_bin = (prev_bin + ALIASING_FACTOR) % MAX_BINS
                im_gradient[:, :, :, prev_bin] = np.add(im_gradient[:, :, :, prev_bin], \
                                                        np.multiply(im_gradient[:, :, :, i], \
                                                        (ALIASING_FACTOR - offset) / ALIASING_FACTOR))
                im_gradient[:, :, :, next_bin] = np.add(im_gradient[:, :, :, next_bin], \
                                                        np.multiply(im_gradient[:, :, :, i], \
                                                        offset / ALIASING_FACTOR))

        # downsample MAX_BINS to NUM_BINS
        im_gradient = im_gradient[:, :, :, ::ALIASING_FACTOR]

    # record histogram of oriented gradients (magnitude)
    im_histogram = np.zeros(((im_rgb.shape[0]-cell_size)//stride+1, \
                                (im_rgb.shape[1]-cell_size)//stride+1, \
                                im_rgb.shape[2], NUM_BINS))
    for row in range(0, height-cell_size+1, stride):
        for col in range(0, width-cell_size+1, stride):
            im_histogram[row//stride, col//stride, :, :] = \
                        np.sum(im_gradient[row:row+cell_size, col:col+cell_size, :, :], axis=(0, 1))
    return im_histogram

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
        "--hog",
        #default="8,8",
        help="Activate HOG Feature Extractor. " +
        "Kernel size, Stride size."
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
    hog_param, mean, channel_swap = None, None, None

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

    if args.hog:
        hog_param = [int(s) for s in args.hog.split(',')]

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

            if hog_param is not None:
                prep_time = 0
                l_start = time.time()
                for i, img in enumerate(inputs):
                    # elem = img.transpose(1, 2, 0).astype(float)
                    # l_start = time.time()
                    # # HOG kernel size : 8x8, stride: 8
                    # ifeat = hog_histogram(elem, hog_param)
                    # # migrate features from 4D to 3D
                    # ifeat = np.reshape(ifeat, (ifeat.shape[0], ifeat.shape[1], -1))
                    # # change row x col x ch -> ch x row x col
                    # if i == 0:
                    #     new_inputs = np.transpose(ifeat, (2, 0, 1))
                    # else:
                    #     new_inputs = np.concatenate((new_inputs, \
                    #                                 np.transpose(ifeat, (2, 0, 1))), axis=0)
                    # prep_time = prep_time + time.time() - l_start

                    elem = img.transpose(1, 2, 0).astype('int16')
                    l_start = time.time()
                    # HOG kernel size : 8x8, stride: 8
                    ifeat = hog_histogram_parallel(elem, hog_param)
                    # migrate features from 4D to 3D
                    ifeat = np.reshape(ifeat, (-1, ifeat.shape[-2], ifeat.shape[-1]))
                    # stack ch x row x col
                    if i == 0:
                        new_inputs = ifeat
                    else:
                        new_inputs = np.concatenate((new_inputs, ifeat), axis=0)
                    prep_time = prep_time + time.time() - l_start

                if len(new_inputs.shape) == 3:
                    inputs = np.reshape(new_inputs, \
                                (1, new_inputs.shape[0], new_inputs.shape[1], new_inputs.shape[2]))
                else:
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
