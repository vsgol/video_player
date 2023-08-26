import json
from argparse import ArgumentParser
from time import perf_counter
from os import listdir
from os.path import isdir, join
import math

import cv2
import numpy as np


class EmptyStudentVideo:
    def swap_images(self):
        pass

    @staticmethod
    def get_image(_):
        return False, None

    def release(self):
        pass

    def cheating(self):
        pass

    def fast_forward(self):
        pass

    def rewind(self):
        pass


class StudentVideo:
    def __init__(self, path, student_name):
        self.path = path
        self.start_cheating = -1
        self.start_google = -1
        self.start_ide = -1
        self.res = []
        self.student_name = student_name
        self.main_capture = cv2.VideoCapture(join(path, student_name, 'webcam.mp4'))
        self.second_capture = cv2.VideoCapture(join(path, student_name, 'window.mp4'))
        self.frame = 0

    def swap_images(self):
        self.main_capture, self.second_capture = self.second_capture, self.main_capture

    def get_image(self, shape):
        ret_main, image_main = self.main_capture.read()
        ret_second, image_second = self.second_capture.read()
        self.frame += 1
        if not ret_second and not ret_main:
            return False, None

        if not ret_second:
            image = cv2.resize(image_main, shape)
        elif not ret_main:
            image = cv2.resize(image_second, shape)
        else:
            image = cv2.resize(image_main, shape)
            image_second = cv2.resize(image_second, (shape[0] // 4, shape[1] // 4))
            image[-shape[1] // 4:, -shape[0] // 4:] = image_second
        if self.start_cheating != -1:
            image[[0, 1, 2, 3, -4, -3, -2, -1], :] = [0, 0, 255]
            image[:, [0, 1, 2, 3, -4, -3, -2, -1]] = [0, 0, 255]
        # if self.start_google != -1:
        #     image[[0, 1, -2, -1], :] = [255, 0, 0]
        #     image[:, [0, 1, -2, -1]] = [255, 0, 0]
        # elif self.start_ide != -1:
        #     image[[0, 1, -2, -1], :] = [0, 255, 0]
        #     image[:, [0, 1, -2, -1]] = [0, 255, 0]

        return True, image

    def release(self):
        self.main_capture.release()
        self.second_capture.release()
        if self.start_cheating != -1:
            self.cheating()
        with open(join(self.path, f"{self.student_name}.json"), 'w') as outfile:
            json.dump({"result": [{"frame_data": self.res}]}, outfile)

    def cheating(self):
        if self.start_cheating != -1:
            self.res.append({
                "start_time": self.start_cheating/30.0,
                "end_time": self.frame/30.0,
                "warn": ["cheating"]
                })
            self.start_cheating = -1
        else:
            self.start_cheating = self.frame

    def fast_forward(self):
        index = max(self.main_capture.get(cv2.CAP_PROP_POS_FRAMES),
                    self.second_capture.get(cv2.CAP_PROP_POS_FRAMES))
        index += 300
        self.main_capture.set(cv2.CAP_PROP_POS_FRAMES, min(index, self.main_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.second_capture.set(cv2.CAP_PROP_POS_FRAMES, min(index, self.second_capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    def rewind(self):
        index = max(self.main_capture.get(cv2.CAP_PROP_POS_FRAMES),
                    self.second_capture.get(cv2.CAP_PROP_POS_FRAMES))
        index -= 300
        self.main_capture.set(cv2.CAP_PROP_POS_FRAMES, max(index, 0))
        self.second_capture.set(cv2.CAP_PROP_POS_FRAMES, max(index, 0))


def player(args):
    split_size = math.ceil(math.sqrt(int(args.number_of_videos)))
    shape = 1720 // split_size, 880 // split_size
    default_fps = 30.0
    fps = default_fps
    pause = False
    selected_window = -1
    video_path = args.input_path
    last_student = 0
    students_names = [f for f in listdir(video_path) if isdir(join(video_path, f))]
    students = [[EmptyStudentVideo()] * split_size for _ in range(split_size)]
    while True:
        start_time = perf_counter()
        if fps > 0 and not pause:
            image = []
            for i in range(split_size):
                image.append([])
                for j in range(split_size):
                    ret, small_window_image = students[i][j].get_image(shape)
                    while not ret:
                        if last_student != len(students_names):
                            students[i][j].release()
                            students[i][j] = StudentVideo(video_path, students_names[last_student])
                            last_student += 1
                            ret, small_window_image = students[i][j].get_image(shape)
                        else:
                            ret, small_window_image = True, np.zeros((shape[1], shape[0], 3))
                    image[i].append(small_window_image)
            image = np.vstack([np.hstack(row) for row in image])

            # Write the resulting frame
            cv2.imshow('Student videos', image.astype(np.uint8))
            work_time = perf_counter() - start_time
            key = cv2.waitKey(int(max(1.0, 1000.0 / fps - work_time)))
        else:
            key = cv2.waitKey(0)

        # Quit
        if key in {ord('q'), ord('Q'), 27}:
            [[student.release() for student in row] for row in students]
            break
        # Swap images
        elif key in {ord('s'), ord('S')}:
            if selected_window != -1:
                students[selected_window // split_size][selected_window % split_size].swap_images()
                selected_window = -1
        # Cheating
        elif key in {ord('c'), ord('C')}:
            if selected_window != -1:
                students[selected_window // split_size][selected_window % split_size].cheating()
                selected_window = -1

        # fast_forward and rewind
        elif key in {ord('l'), ord('L')}:
            if selected_window != -1:
                students[selected_window // split_size][selected_window % split_size].fast_forward()
        elif key in {ord('j'), ord('J')}:
            if selected_window != -1:
                students[selected_window // split_size][selected_window % split_size].rewind()

        # select video
        elif ord('1') <= key <= ord(args.number_of_videos):
            selected_window = key - ord('1')

        # fps
        elif key == ord('+'):
            fps += 0.25 * default_fps
            fps = min(default_fps * 5, fps)
        elif key == ord('-'):
            fps -= 0.25 * default_fps
            fps = max(0.0, fps)
        elif key == ord('='):
            fps = default_fps
        elif key == ord(' ') or key == ord('k') or key == ord('K'):
            pause = not pause

    cv2.destroyWindow('Student videos')


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('input_path', help='Required. An input to process. The input must be a folder with video')
    parser.add_argument('number_of_videos', help='Number of videos that will be shown on the screen at the same time '
                                                 '(from 1 to 9)')
    return parser


if __name__ == '__main__':
    print('''
Controls
  Space/K: Play/pause, +: Faster, -: Slower, =: Normal rate, 
  J: rewind 10 seconds, L: fast forward 10 seconds, 
  `number`+S: Swap images of `number` student, 
  `number`+C: Student `number` began/finished cheating'''.lstrip())
    args = build_argparser().parse_args()
    player(args)
