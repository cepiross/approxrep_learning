#!/usr/bin/env python3
"""
Author: jpc5731@cse.psu.edu (Jinhang Choi), 09/18/2017

cvtlmdb_img2hog.py is an lmdb converter callable from the command line.
By default it changes raw-image lmdb to HOG-based feature lmdb.
Usage : ./cvtlmdb_img2hog.py --input_file inlmdb_path --outputfile outlmdb_path
"""
import os
import sys
import argparse
import math
import cv2
import numpy as np
import lmdb

CAFFE_PYTHON = '/specify/your/cafe/python/path'
sys.path.insert(0, CAFFE_PYTHON)
import caffe

MAP_SIZE = 1099511627776 # 1TiB
RESUME_KEY = ''
#RESUME_KEY = 'specify your lmdb element key to be resumed'.encode('ascii')

# I am going to try anti-aliasing according to indication of finer degrees
NUM_BINS = 3
ALIASING_FACTOR = 3
MAX_BINS = NUM_BINS * ALIASING_FACTOR
UNIT_DEGREE = 2 * math.pi / MAX_BINS
TRIGON_PAIRS = np.zeros((2, MAX_BINS))
for idx in range(MAX_BINS):
    TRIGON_PAIRS[0][idx] = math.cos(idx * UNIT_DEGREE)
    TRIGON_PAIRS[1][idx] = math.sin(idx * UNIT_DEGREE)

def hog_histogram(im_rgb, cell_size, stride):
    '''Make cell-wise histogram of oriented gradients'''
    height = im_rgb.shape[0]
    width = im_rgb.shape[1]

    # start to compute element-wise gradient, magnitude, and max orientation
    # according to Ix * cos(theta) + Iy * sin(theta) where theta = {0, ..., 2*pi/max_bins}
    im_ix = cv2.filter2D(im_rgb, -1, np.array([[-1, 0, 1]]))
    im_iy = cv2.filter2D(im_rgb, -1, np.array([[-1], [0], [1]]))
    im_gradient = np.zeros((im_rgb.shape[0], im_rgb.shape[1], im_rgb.shape[2], MAX_BINS))
    for i in range(MAX_BINS):
        im_gradient[:, :, :, i] = im_ix[:, :, :] * TRIGON_PAIRS[0][i] + \
                                  im_iy[:, :, :] * TRIGON_PAIRS[1][i]
    im_gradmax = np.max(im_gradient, axis=3)
    im_orientation = np.argmax(im_gradient, axis=3)

    # spread magnitude with respect to orientation theta
    # such that argmax Ix*cos+Iy*sin = Msin(theta-alpha) where theta = alpha
    im_gradient = np.zeros_like(im_gradient)
    for i in range(MAX_BINS):
        loc = np.where(im_orientation == i)
        im_gradient[loc[0], loc[1], loc[2], i] = im_gradmax[loc[0], loc[1], loc[2]]

    # bilinear interploation of magnitude to mitigate aliasing
    for i in range(MAX_BINS):
        offset = i % ALIASING_FACTOR
        if offset != 0:
            prev_bin = i - offset
            next_bin = (prev_bin + ALIASING_FACTOR) % MAX_BINS
            im_gradient[:, :, :, prev_bin] += im_gradient[:, :, :, i] * \
                                                (ALIASING_FACTOR - offset) / ALIASING_FACTOR
            im_gradient[:, :, :, next_bin] += im_gradient[:, :, :, i] * \
                                                offset / ALIASING_FACTOR
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

def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments: input file.
    parser.add_argument(
        "--input_file",
        help="Input data file."
    )
    parser.add_argument(
        "--output_file",
        help="Output_data_file."
    )

    args = parser.parse_args()
    if not args.input_file:
        print("No Input!")
        return
    if not args.output_file:
        print("No Output!")
        return

    rddb = lmdb.open(args.input_file, readonly=True)
    wrdb = lmdb.open(args.output_file, map_size=MAP_SIZE)
    datum = caffe.proto.caffe_pb2.Datum()

    resume_flag = bool(RESUME_KEY and RESUME_KEY.strip())
    with rddb.begin() as rdtxn:
        cursor = rdtxn.cursor()
        for key, value in cursor:
            if resume_flag is True:
                print (key, 'Skip')
                if key == RESUME_KEY:
                    resume_flag = False
                continue
            datum.ParseFromString(value)
            label = datum.label
            # change ch x row x col -> row x col x ch
            img = np.transpose(caffe.io.datum_to_array(datum), (1, 2, 0))
            img = img.astype(float)
            # vanishing problem due to small value near to 0?
            ifeat = hog_histogram(img, 8, 8)
            # migrate features from 5D to 3D
            ifeat = np.reshape(ifeat, (ifeat.shape[0], ifeat.shape[1], -1))
            # change row x col x ch -> ch x row x col
            tmp = np.transpose(ifeat, (2, 0, 1))
            data = caffe.io.array_to_datum(tmp, label)

            with wrdb.begin(write=True) as wrtxn:
                if wrtxn.put(key, data.SerializeToString()) is False:
                    print ('Error!')
                    print (wrdb.info())
                    return
                else:
                    print (key, data.channels, data.height, data.width, 'Done')

if __name__ == '__main__':
    main(sys.argv)
