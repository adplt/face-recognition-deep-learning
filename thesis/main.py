import argparse
import importlib

EYES = [i for i in range(36, 48)]


def main(argv):
    align_data = importlib.import_module('align_data')
    align_data.align_face_lfw(argv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--shapePredictor', required=True, help='Path to facial landmark')
    parser.add_argument('-i', '--image', required=False, help='Path to Input Image')
    args = parser.parse_args()
    main(args)

# Example:
# python ./thesis/main.py --shapePredictor shape_predictor_68_face_landmarks.dat
