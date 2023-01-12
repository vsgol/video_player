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
    def get_image(shape):
        return False, None

    def release(self):
        pass

    def cheating(self):
        pass


class StudentVideo:
    def __init__(self, path, student_name):
        self.is_cheating = False
        self.output = open(join(path, student_name, 'cheating_frames.txt'), 'a')
        self.student_name = student_name
        self.webcam_capture = cv2.VideoCapture(join(path, student_name, 'webcam.webm'))
        self.window_capture = cv2.VideoCapture(join(path, student_name, 'window.webm'))
        self.frame = 0

    def swap_images(self):
        self.webcam_capture, self.window_capture = self.window_capture, self.webcam_capture

    def get_image(self, shape):
        ret_webcam, image_webcam = self.webcam_capture.read()
        ret_window, image_window = self.window_capture.read()
        self.frame += 1
        if not ret_window:
            return ret_webcam, image_webcam
        if not ret_webcam:
            return True, image_window
        image_webcam = cv2.resize(image_webcam, shape)
        image_window = cv2.resize(image_window, (shape[0] // 4, shape[1] // 4))
        image_webcam[-shape[1] // 4:, -shape[0] // 4:] = image_window
        if self.is_cheating:
            image_webcam[[0, 1, -2, -1], :] = [0, 0, 255]
            image_webcam[:, [0, 1, -2, -1]] = [0, 0, 255]
        return True, image_webcam

    def release(self):
        self.webcam_capture.release()
        self.window_capture.release()
        if self.is_cheating:
            self.cheating()

    def cheating(self):
        if self.is_cheating:
            self.output.write(f' - {self.frame}\n')
        else:
            self.output.write(self.frame.__str__())
        self.is_cheating = not self.is_cheating


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('input_path', help='Required. An input to process. The input must be a folder with video')
    parser.add_argument('number_of_videos', help='Number of videos that will be shown on the screen at the same time '
                                                 '(from 1 to 9)')
    return parser


def main():
    print('Controls')
    print('Space: Play/pause, +: Faster, -: Slower, =: Normal rate, number+s: Swap images of `number` student' +
          'number+c: Student `number` began/finished cheating')
    args = build_argparser().parse_args()
    split_size = math.ceil(math.sqrt(int(args.number_of_videos)))
    shape = 1920 // split_size, 1080 // split_size
    default_fps = 30.0
    fps = default_fps
    selected_window = -1
    video_path = args.input_path
    last_student = 0
    students_names = [f for f in listdir(video_path) if isdir(join(video_path, f))]
    students = [[EmptyStudentVideo()] * split_size for _ in range(split_size)]
    while True:
        start_time = perf_counter()
        if fps > 0:
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
                            ret, small_window_image = True, np.zeros(shape)
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
            break
        # Swap images
        elif key in {ord('s'), ord('S')}:
            if selected_window != -1:
                students[selected_window // split_size][selected_window % split_size].swap_images()
                selected_window = -1
        elif key in {ord('c'), ord('c')}:
            if selected_window != -1:
                students[selected_window // split_size][selected_window % split_size].cheating()
                selected_window = -1
        elif ord('1') <= key <= ord(args.number_of_videos):
            selected_window = key - ord('1')
        elif key == ord('+'):
            fps += 0.25 * default_fps
            fps = min(default_fps * 3, fps)
        elif key == ord('-'):
            fps -= 0.25 * default_fps
            fps = max(0.0, fps)
        elif key == ord('='):
            fps = default_fps
    cv2.destroyWindow('Student videos')


if __name__ == '__main__':
    main()
