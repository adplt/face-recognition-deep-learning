from imutils.face_utils import FaceAligner
from PIL import Image

import imutils, dlib, cv2, os, inspect, glob, shutil, numpy


def align_face(args):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=64, desiredFaceHeight=64)

    # load the input image, resize it, and convert it to gray scale
    curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    input_dir = os.path.join(curr_directory, 'input_dir')
    out_dir = os.path.join(curr_directory, 'out_dir')
    list_label = os.listdir(input_dir)

    i = 0

    while i < len(list_label):
        j = 0
        label = list_label[i]
        label_dir = os.listdir(os.path.join(input_dir, label))
        while j < len(label_dir):
            image = cv2.imread(os.path.join(input_dir, label + '/' + label_dir[j]))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # show the original input image and detect faces in the grayscale
            rects = detector(gray, 2)

            # loop over the face detections
            for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                # (x, y, w, h) = rect_to_bb(rect)
                # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
                face_aligned = fa.align(image, gray, rect)

                # display the output images
                # cv2.imshow('Aligned', faceAligned)
                # cv2.waitKey(0)

                img = Image.fromarray(face_aligned, 'RGB')
                img.save(label_dir[j])

                if not os.path.exists(os.path.join(out_dir, label)):
                    os.makedirs(os.path.join(out_dir, label))
                # shutil.copy(img, os.path.join(out_dir, label))
            j += 1
        i += 1
