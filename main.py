import create_dir
import align_data
from six.moves import urllib
import sys
import os
import tarfile
import argparse
from tqdm import tqdm
import requests
import shutil


TRAINING_DATA = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
# TRAINING_DATA = 'http://www.security-camera-warehouse.com/images/lens-distance/50ft/3MP-12mm.png'


file_name = TRAINING_DATA.split('/')[-1]
folder_name = file_name[:file_name.index('.')]

chunk_size = 1024


def main(argv):
    create_dir
    r = requests.get(TRAINING_DATA, stream=True)
    if r.headers.get('content-length') is not None:
        size = int(r.headers.get('content-length'))
    else:
        size = 0
    if not os.path.exists('input_dir') and not os.path.isdir('input_dir') and not os.path.exists('lfw.tgz'):
        r = requests.get(TRAINING_DATA, stream=True)
        with open(file_name, 'wb') as f:
            for data in tqdm(iterable=r.iter_content(chunk_size=chunk_size), total=size / chunk_size, unit='KB'):f.write(data)
        print 'Successfully downloaded ' + file_name + ' with ' + str(size / 1000000) + ' MB'
    if os.path.exists('input_dir') and os.path.exists('lfw.tgz'):
        print 'Extracting ' + file_name + ' ...'
        tarfile.open(file_name).extractall()
        os.rename(folder_name, 'input_dir')
        os.remove(file_name)
        print 'Successfully extracted ' + file_name + ' to input_dir and remove ' + file_name
    else:
        print 'All the input data have been complete'
    align_data.align_face(argv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--shapePredictor', required=False, help='Path to facial landmark')
    parser.add_argument('-i', '--image', required=False, help='Path to Input Image')
    args = parser.parse_args()
    main(args)


# Example:
# python main.py --shape-predictor shape_predictor_68_face_landmarks.dat --image input_dir/Aaron_Eckhart/Aaron_Eckhart_0001.jpg
