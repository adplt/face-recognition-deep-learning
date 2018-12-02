import align_data
import argparse


def main(argv):
    align_data.align_face_feret_color(argv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--shapePredictor', required=False, help='Path to facial landmark')
    parser.add_argument('-i', '--image', required=False, help='Path to Input Image')
    args = parser.parse_args()
    main(args)

# Example:
# python main.py --shape-predictor shape_predictor_68_face_landmarks.dat
