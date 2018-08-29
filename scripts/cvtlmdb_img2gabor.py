#!/usr/bin/env python3
"""
Author: jpc5731@cse.psu.edu (Jinhang Choi), 09/18/2017

cvtlmdb_img2gabor.py is an lmdb converter callable from the command line.
By default it changes raw-image lmdb to gabor-based feature lmdb.
Usage : ./cvtlmdb_img2gabor.py --input_file inlmdb_path --outputfile outlmdb_path
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

CELL_SIZE = (8, 8)
# Since there is no difference between clockwise/anti-clockwise degree, 
# I am going to just check 0 ~ 180 degrees
MAX_BINS = 3
SIGMA = (2.0, 8.0)   # gaussian range (smaller -> circle, larger -> line)
UNIT_DEGREE = math.pi / MAX_BINS
LAMBDA = (8.0,) # thickness
GAMMA = 1      # x' versus y'
PSI = 0        # center

# gabor_complex   = exp(-(x'^2+gamma^2*y'^2)/(2*sigma^2))*exp(i(2*pi*x'/lambda + psi))
# gabor_real      = exp(-(x'^2+gamma^2*y'^2)/(2*sigma^2))*cos(2*pi*x'/lambda + psi)
# gabor_imaginary = exp(-(x'^2+gamma^2*y'^2)/(2*sigma^2))*sin(2*pi*x'/lambda + psi)
# such that
#   x' = x*cos(theta) + y*sin(theta)
#   y' = -x*sin(theta) + y*cos(theta)
GABORS = []
for sigma in SIGMA:
    for thick in LAMBDA:
        for bins in range(MAX_BINS):
            # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
            #   ksize - size of gabor filter (n, n)
            #   sigma - standard deviation of the gaussian function
            #   theta - orientation of the normal to the parallel stripes
            #   lambda - wavelength of the sunusoidal factor
            #   gamma - spatial aspect ratio
            #   psi - phase offset
            #   ktype - type and range of values that each pixel in the gabor kernel can hold
            GABORS.append(cv2.getGaborKernel(CELL_SIZE, sigma, bins * UNIT_DEGREE, thick, GAMMA, PSI, ktype=cv2.CV_32F))


def gabor_histogram(im_rgb, cell_size, stride):
    '''Make cell-wise histogram of oriented gradients'''
    height = im_rgb.shape[0]
    width = im_rgb.shape[1]
    channels = im_rgb.shape[2]

    # start to convolve gabor filter according to theta = {0, pi/max_bins, ..., pi*(1-1/max_bins)}
    out_height = (height - cell_size[0])//stride + 1
    out_width = (width - cell_size[1])//stride + 1
    im_features = np.zeros((out_height, out_width, channels, len(GABORS)))
    #cv2.imshow('orig', im_rgb/255.0)
    for idx, gabor in enumerate(GABORS):
        im_features[:, :, :, idx] = cv2.filter2D(im_rgb, -1, gabor)[cell_size[0]//2:height:stride, cell_size[1]//2:height:stride, :]

        #maxval = np.amax(im_features[:, :, :, idx])

        #cv2.imshow('filter', gabor)
        #cv2.imshow('image', im_features[:, :, :, idx]/maxval)
        #cv2.waitKey(0)

    return im_features

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
            ifeat = gabor_histogram(img, CELL_SIZE, 8)
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
