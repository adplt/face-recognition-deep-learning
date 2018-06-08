import create_dir
import align_data
from six.moves import urllib
import sys
import os
import tarfile
import argparse


TRAINING_DATA = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'

file_name = TRAINING_DATA.split('/')[-1]
folder_name = file_name[:file_name.index('.')]


def main(argv):
    create_dir
    # if not os.path.exists('input_dir'):
    #     def download_progress(count, block_size, total_size):
    #         sys.stdout.write('Downloading %s %.1f%% \n' %
    #                          (file_name, float(count * block_size) / float(total_size) * 100.00))
    #         sys.stdout.flush()
    #     info, _ = urllib.request.urlretrieve(TRAINING_DATA, file_name, download_progress)
    #     status = os.stat(info)
    #     print 'Successfully downloaded ' + file_name + ' with ' + str(status.st_size / 1000000) + ' MB'
    # print 'Extracting ' + file_name + ' ...'
    # tarfile.open(file_name).extractall()
    # os.rename(folder_name, 'input_dir')
    # os.remove(file_name)
    # print 'Successfully extracted ' + file_name + ' to input_dir and remove ' + file_name
    align_data.align_face(argv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--shapePredictor', required=True, help='Path to facial landmark')
    parser.add_argument('-i', '--image', required=True, help='Path to Input Image')
    args = parser.parse_args()
    main(args)


# Example:
# python main.py --shape-predictor shape_predictor_68_face_landmarks.dat --image input_dir/Aaron_Eckhart/Aaron_Eckhart_0001.jpg