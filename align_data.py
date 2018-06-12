from imutils.face_utils import FaceAligner, rect_to_bb
from PIL import Image

import imutils, dlib, cv2, os, inspect, glob, shutil, numpy, png


def align_face(args):
    curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    input_dir = os.path.join(curr_directory, 'input_dir')
    out_dir = os.path.join(curr_directory, 'out_dir')
    list_label = os.listdir(input_dir)

    if os.path.exists(out_dir) and len(os.listdir(out_dir)) <= 1:
        print 'Face Aligning ...'

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor and the face aligner
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        fa = FaceAligner(predictor, desiredFaceWidth=64, desiredFaceHeight=64)

        # load the input image, resize it, and convert it to gray scale
        i = 0
        while i < len(list_label):
            j = 0
            label = list_label[i]
            input_label_dir = os.listdir(os.path.join(input_dir, label))
            while j < len(input_label_dir):
                image = cv2.imread(os.path.join(input_dir, label + '/' + input_label_dir[j]))

                # Bicubic Interpolation: extension dari cubic interpolation, membuat permukaan gambar ljadi lebih lembut
                # image = cv2.resize(img, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_CUBIC)

                # detect faces in the grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Changing Color Space - COLOR_BGR2GRAY
                rects = detector(gray, 3)

                # loop over the face detections
                for rect in rects:
                    # extract the ROI of the *original* face, then align the face
                    # using facial landmarks
                    # (x, y, w, h) = rect_to_bb(rect)
                    # face_ori = imutils.resize(image[y:y + h, x:x + w], width=256)
                    face_aligned = fa.align(image, gray, rect)

                    # display the output images
                    # cv2.imshow('Aligned', face_aligned)
                    # cv2.waitKey(0)
                output_label_dir = os.path.join(out_dir, label)
                if not os.path.exists(output_label_dir):
                    os.makedirs(os.path.join(out_dir, label))
                cv2.imwrite(os.path.join(output_label_dir, input_label_dir[j]), face_aligned)
                j += 1
            i += 1
    print 'Successfully face aligned and copy new dataset to to output_dir'
