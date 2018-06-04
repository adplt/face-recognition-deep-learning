import create_dir
from six.moves import urllib
import sys
import os
import tarfile


TRAINING_DATA = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'

file_name = TRAINING_DATA.split('/')[-1]
folder_name = file_name[:file_name.index('.')]

print folder_name


def main(args):
    create_dir
    if not os.path.exists('input_dir'):
        def download_progress(count, block_size, total_size):
            sys.stdout.write('Downloading %s %.1f%% \n' %
                             (file_name, float(count * block_size) / float(total_size) * 100.00))
            sys.stdout.flush()
        info, _ = urllib.request.urlretrieve(TRAINING_DATA, file_name, download_progress)
        status = os.stat(info)
        print 'Successfully downloaded ' + file_name + ' with ' + str(status.st_size / 1000000) + ' MB'
    print 'Extracting ' + file_name + ' ...'
    tarfile.open(file_name).extractall()
    os.rename(folder_name, 'input_dir')
    os.remove(file_name)
    print 'Successfully extracted ' + file_name + ' to input_dir and remove ' + file_name


if __name__ == '__main__':
    main(sys.argv[1:])
