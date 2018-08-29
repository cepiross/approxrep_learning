#!/usr/bin/env python3
"""
Author: jpc5731@cse.psu.edu (Jinhang Choi), 09/18/2017

cvtlmdb_img2resize.py is an lmdb converter callable from the command line.
By default it resizes raw-image lmdb by resize ratio. (e.g 256x256 -> 32x32)
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
RESIZE_RATIO = 8

# I am going to try aliasing according to indication of finer degrees
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
            height, width = img.shape[:2]
            ifeat = cv2.resize(img, (height//RESIZE_RATIO, width//RESIZE_RATIO))
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
