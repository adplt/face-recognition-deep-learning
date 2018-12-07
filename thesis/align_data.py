from imutils.face_utils import FaceAligner
import dlib
import cv2
import os
import inspect


def align_face_lfw(args):
    curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    input_dir = os.path.join(curr_directory, 'input_dir')
    out_dir = os.path.join(curr_directory, 'out_dir')
    list_label = os.listdir(input_dir)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) <= 1 and args.shapePredictor is not None:
        print('Face Aligning ...')
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join(curr_directory, args.shapePredictor))
        fa = FaceAligner(predictor, desiredFaceWidth=24, desiredFaceHeight=24)
        i = 0
        while i < len(list_label):
            j = 0
            label = list_label[i]
            input_label_dir = os.listdir(os.path.join(input_dir, label))
            output_label_dir = os.path.join(out_dir, label)
            face_aligned = None
            while j < len(input_label_dir):
                image = cv2.imread(os.path.join(input_dir, label + '/' + input_label_dir[j]))
                # Bicubic Interpolation: extension dari cubic interpolation, membuat permukaan gambar jadi lebih lembut
                # tuple dapat diisi dengan None (size'a bakal ngikutin yg default dari OpenCV)
                image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                rects = detector(gray, 3)
                for rect in rects:
                    face_aligned = fa.align(image, gray, rect)
                if not os.path.exists(output_label_dir):
                    os.makedirs(output_label_dir)
                    cv2.imwrite(os.path.join(output_label_dir, input_label_dir[j]), face_aligned)
                j += 1
            i += 1
        print('Successfully face aligned and copy new dataset to to output_dir')
    elif args.shapePredictor is None:
        print('You don\'t have shapePredictor so you cannot continuing to alignment face')
    else:
        print('')


def align_face_youtube_face(args):
    curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    input_dir = os.path.join(curr_directory, 'input_dir')
    out_dir = os.path.join(curr_directory, 'out_dir')
    list_label = os.listdir(input_dir)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) <= 1 and args.shapePredictor is not None:
        print('Face Aligning ...')
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join(curr_directory, args.shapePredictor))
        fa = FaceAligner(predictor, desiredFaceWidth=24, desiredFaceHeight=24)
        i = 0
        while i < len(list_label):
            j = 0
            label_path_in = os.path.join(input_dir, list_label[i])
            label_path_out = os.path.join(out_dir, list_label[i])
            frame_index_list = os.listdir(label_path_in)
            face_aligned = None
            while j < len(frame_index_list):
                k = 0
                frame_index_path = os.path.join(label_path_in, frame_index_list[j])
                if frame_index_path is not '.DS_Store':
                    input_label_dir = os.listdir(frame_index_path)
                    # print 'label_path_in: ' + frame_index_list[j]
                    # print 'frame_index_path: ' + frame_index_path
                    # print 'input_label_dir: ' + str(len(input_label_dir))
                    # print 'j: ' + str(j)
                    # print '\n\n\n\n'
                    while k < len(input_label_dir):
                        # print 'k: ' + str(k)
                        # print 'frame_index_path: ' + frame_index_path
                        # print 'input_label_dir[k]: ' + input_label_dir[k]
                        # print 'os.path.join(frame_index_path, input_label_dir[k]): ' +\
                        #       os.path.join(frame_index_path, input_label_dir[k])
                        # print '\n'
                        if input_label_dir[k] is not '.DS_Store':
                            image = cv2.imread(os.path.join(frame_index_path, input_label_dir[k]))
                            if image is None:
                                print('image failed: ' + os.path.join(frame_index_path, input_label_dir[k]))
                            elif image is not None:
                                # Bicubic Interpolation:
                                # extension dari cubic interpolation, membuat permukaan gambar jadi lebih lembut
                                # tuple dapat diisi dengan None (size'a bakal ngikutin yg default dari OpenCV)
                                # a bicubic interpolation over 4x4 pixel neighborhood
                                image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        
                                gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                rects = detector(gray, 3)
        
                                for rect in rects:
                                    face_aligned = fa.align(image, gray, rect)
                                if not os.path.exists(label_path_out):
                                    os.makedirs(label_path_out)
                                    cv2.imwrite(os.path.join(label_path_out, str(input_label_dir[j])), face_aligned)
                                else:
                                    cv2.imwrite(os.path.join(label_path_out, input_label_dir[k]), face_aligned)
                        k += 1;
                j += 1;
            i += 1;
        print('Successfully face aligned and copy new dataset to to output_dir')
    elif args.shapePredictor is None:
        print('You don\'t have shapePredictor so you cannot continuing to alignment face')
    else:
        print('')


def align_face_feret_color(args):
    curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    input_dir = os.path.join(curr_directory, 'input_dir')
    out_dir = os.path.join(curr_directory, 'out_dir')
    list_label = os.listdir(input_dir)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) <= 1 and args.shapePredictor is not None:
        print('Face Aligning ...')
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join(curr_directory, args.shapePredictor))
        fa = FaceAligner(predictor, desiredFaceWidth=24, desiredFaceHeight=24)
        i = 0
        while i < len(list_label):
            print("i: " + str(i) + " " + list_label[i] is not '.DS_Store' + " " + list_label[i])
            if list_label[i] is not '.DS_Store':
                dataset_index = list_label[i]
                file = open(os.path.join(dataset_index, str(dataset_index + ".xml")), "r")
            
                print("file: " + file)
            i += 1
            
            # input_label_dir = os.listdir(os.path.join(input_dir, label))
            # output_label_dir = os.path.join(out_dir, label)
            # face_aligned = None
            # while j < len(input_label_dir):
            #     image = cv2.imread(os.path.join(input_dir, label + '/' + input_label_dir[j]))
            # Bicubic Interpolation: extension dari cubic interpolation, membuat permukaan gambar jadi lebih lembut
            # tuple dapat diisi dengan None (size'a bakal ngikutin yg default dari OpenCV)
            #     image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            #
            #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #     rects = detector(gray, 3)
            #     for rect in rects:
            #         face_aligned = fa.align(image, gray, rect)
            #     if not os.path.exists(output_label_dir):
            #         os.makedirs(output_label_dir)
            #         cv2.imwrite(os.path.join(output_label_dir, input_label_dir[j]), face_aligned)
            #     j += 1
            # i += 1
        print('Successfully face aligned and copy new dataset to to output_dir')
    elif args.shapePredictor is None:
        print('You don\'t have shapePredictor so you cannot continuing to alignment face')
    else:
        print('')

