from imutils.face_utils import FaceAligner
import dlib
import cv2
import os
import inspect
import bz2
import re

LEFT_EYE = re.compile('leftEye x=' + '"' + '[0-9]+')
RIGHT_EYE = re.compile('rightEye x=' + '"' + '[0-9]+')
PERSON_ID = re.compile('person id=' + '"' + '[0-9]+')
FRAME_ID = re.compile('frame number=' + '"' + '[0-9]+')
REG_NUM = re.compile('[0-9]+')


class PreProcessing:
    
    def __init__(self, args):
        self.args = args
        print('Pre Processing')
    
    def align_face_lfw(self):
        curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_dir = os.path.join(curr_directory, 'input_dir')
        out_dir = os.path.join(curr_directory, 'out_dir')
        list_label = os.listdir(input_dir)
        assert os.path.exists(out_dir) and len(os.listdir(out_dir)) <= 1 and self.args.shapePredictor is not None
        print('Face Aligning ...')
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join(curr_directory, self.args.shapePredictor))
        fa = FaceAligner(predictor, desiredFaceWidth=24, desiredFaceHeight=24)
        i = 0
        while i < len(list_label):
            j = 0
            label = list_label[i]
            input_label_dir = os.listdir(os.path.join(input_dir, label))
            output_label_dir = os.path.join(out_dir, label)
            while j < len(input_label_dir):
                if os.path.isdir(os.path.join(input_dir, label + '/' + input_label_dir[j])):
                    input_label_dir_child = os.listdir(os.path.join(input_dir, label + '/' + input_label_dir[j]))
                    k = 0
                    while k < len(input_label_dir_child):
                        image = cv2.imread(os.path.join(input_dir, label + '/' + input_label_dir[j] + '/' + input_label_dir_child[k]))
                        # Bicubic Interpolation: extension dari cubic interpolation, membuat permukaan gambar jadi lebih lembut
                        # tuple dapat diisi dengan None (size'a bakal ngikutin yg default dari OpenCV)
                        # a bicubic interpolation over 4x4 pixel neighborhood
                        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        rects = detector(gray, 3)
                        for rect in rects:
                            face_aligned = fa.align(image, gray, rect)
                            if not os.path.exists(output_label_dir):
                                os.makedirs(output_label_dir)
                            cv2.imwrite(os.path.join(output_label_dir, input_label_dir_child[k]), face_aligned)
                        k += 1
                else:
                    image = cv2.imread(os.path.join(input_dir, label + '/' + input_label_dir[j]))
                    # Bicubic Interpolation: extension dari cubic interpolation, membuat permukaan gambar jadi lebih lembut
                    # tuple dapat diisi dengan None (size'a bakal ngikutin yg default dari OpenCV)
                    # a bicubic interpolation over 4x4 pixel neighborhood
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
    
    def align_face_youtube_face(self):
        curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_dir = os.path.join(curr_directory, 'input_dir')
        out_dir = os.path.join(curr_directory, 'out_dir_1')
        list_label = os.listdir(input_dir)
        assert os.path.exists(out_dir) and len(os.listdir(out_dir)) <= 1 and self.args.shapePredictor is not None
        print('Face Aligning ...')
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join(curr_directory, self.args.shapePredictor))
        fa = FaceAligner(predictor, desiredFaceWidth=24, desiredFaceHeight=24)
        i = 0
        while i < len(list_label):
            j = 0
            label_path_in = os.path.join(input_dir, list_label[i])
            label_path_out = os.path.join(out_dir, list_label[i])
            frame_index_list = os.listdir(label_path_in)
            while j < len(frame_index_list):
                k = 0
                frame_index_path = os.path.join(label_path_in, frame_index_list[j])
                if frame_index_path is not '.DS_Store':
                    input_label_dir = os.listdir(frame_index_path)
                    while k < len(input_label_dir):
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
                                    cv2.imwrite(os.path.join(label_path_out, input_label_dir[k]), face_aligned)
                        k += 1;
                j += 1;
            i += 1;
        print('Successfully face aligned and copy new dataset to to output_dir')
     
    def align_face_choke_point(self):
        curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_dir = os.path.join(curr_directory, 'input_dir')
        out_dir = os.path.join(curr_directory, 'out_dir')
        list_label = os.listdir(input_dir)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join(curr_directory, self.args.shapePredictor))
        fa = FaceAligner(predictor, desiredFaceWidth=24, desiredFaceHeight=24)
        assert os.path.exists(out_dir) and len(os.listdir(out_dir)) <= 1 and self.args.shapePredictor is not None
        print('Face Aligning ...')
        i = 0
        while i < len(list_label):
            if list_label[i] is not '.DS_Store':
                dataset_index = list_label[i]
                images = os.path.join(input_dir, dataset_index)
                file = open(os.path.join(input_dir, dataset_index, str(dataset_index + ".xml")), "r")
                left_x = 0
                left_y = 0
                right_x = 0
                right_y = 0
                save_frame_id = 0
                save_person_id = 0
                j = 0
                for line in file.readlines():
                    person_id = re.findall(PERSON_ID, line)
                    left = re.findall(LEFT_EYE, line)
                    right = re.findall(RIGHT_EYE, line)
                    frame_id = re.findall(FRAME_ID, line)
                    if len(frame_id) > 0:
                        number = re.findall(REG_NUM, line)
                        save_frame_id = number[0]
                    elif len(person_id) > 0:
                        id = re.findall(REG_NUM, line)
                        save_person_id = id[0]
                        label_path_out = os.path.join(out_dir, str(save_person_id))
                    elif len(left) > 0:
                        x, y = re.findall(REG_NUM, line)
                        left_x = x
                        left_y = y
                    elif len(right) > 0:
                        x, y = re.findall(REG_NUM, line)
                        right_x = x
                        right_y = y
                    else:
                        if save_frame_id != 0 and save_person_id != 0 and left_x != 0 and left_y != 0 and right_x != 0 and right_y != 0:
                            image = cv2.imread(os.path.join(images, str(save_frame_id) + '.jpg'))
                            # img = cv2.rectangle(
                            #     image,
                            #     (int(left_x), int(left_y)),
                            #     (int(right_x), int(right_y)),
                            #     (255, 0, 0),
                            #     2
                            # )
                            img_resize = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                            gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2BGR)
                            rects = detector(gray, 3)
                            for rect in rects:
                                face_aligned = fa.align(img_resize, gray, rect)
                                if not os.path.exists(label_path_out):
                                    os.makedirs(label_path_out)
                                cv2.imwrite(os.path.join(label_path_out, str(j) + '.png'), face_aligned)
                                j += 1
                                # cv2.imshow(str(save_person_id), face_aligned)
                                # cv2.waitKey(30)
            i += 1
        print('Successfully face aligned and copy new dataset to to output_dir')
    
    def align_face_feret_color(self):
        curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_dir = os.path.join(curr_directory, 'input_dir')
        out_dir = os.path.join(curr_directory, 'out_dir')
        list_label = os.listdir(input_dir)
        assert os.path.exists(out_dir) and len(os.listdir(out_dir)) <= 1 and self.args.shapePredictor is not None
        print('Face Aligning ...')
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join(curr_directory, self.args.shapePredictor))
        fa = FaceAligner(predictor, desiredFaceWidth=24, desiredFaceHeight=24)
        i = 0
        while i < len(list_label):
            j = 0
            label = list_label[i]
            input_label_dir = os.listdir(os.path.join(input_dir, label))
            output_label_dir = os.path.join(out_dir, label)
            while j < len(input_label_dir):
                
                old_name = os.path.join(input_dir, label + '/' + input_label_dir[j])
                new_name = os.path.join(input_dir, label + '/' + input_label_dir[j][0:len(input_label_dir[j])-4])
    
                print old_name
                
                dec = bz2.BZ2Decompressor()
                with open(old_name, "rb") as fobj:
                    data = fobj.read()
    
                data = dec.decompress(data)
                
                print data
                
                image = cv2.imread(new_name)
                
                # Bicubic Interpolation: extension dari cubic interpolation, membuat permukaan gambar jadi lebih lembut
                # tuple dapat diisi dengan None (size'a bakal ngikutin yg default dari OpenCV)
                # a bicubic interpolation over 4x4 pixel neighborhood
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
