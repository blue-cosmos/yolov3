# -*- coding: utf-8 -*-
import time
import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage, QCloseEvent
from PyQt5.QtWidgets import QDialog, QMessageBox, QApplication
from PyQt5.uic import loadUi
import cv2 as cv
import multiprocessing as mp
import argparse

import darknet
from detect import detect as yolov3_detect

def get_prog_config(config_file_path: str, section:str) ->dict:
    import configparser
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return config[section]

class CountdownStudy(QDialog):
    def __init__(self, *args):
        super().__init__(*args)
        cwd = os.path.dirname(__file__)
        os.environ['PATH'] = cwd + ';' + os.environ['PATH']
        ui_file_path = os.path.join(cwd, "countdown_study.ui")
        loadUi(ui_file_path, self)

        self.camera_label.setScaledContents(True)

        self._duration_buttons = ((self.one_min_button, 1),
                   (self.three_min_button, 3),
                   (self.five_min_button, 5))
        for button, _ in self._duration_buttons:
            button.toggled.connect(self.show_initial_lcd_coundown)
        self.one_min_button.setChecked(True)
        
        self.action_button.clicked.connect(self.action_start)

        self.video_worker = VideoWorker()
        self.video_worker.image_ready_signal[QPixmap].connect(self.show_video_image)
        self.video_worker.detection_signal[list].connect(self.obj_detected)
        self.video_worker.start()

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.video_worker.stop()
        return super().closeEvent(a0)

    def show_initial_lcd_coundown(self):
        for button, duration in self._duration_buttons:
            if button.isChecked():
                self.countdown_lcd.display(f'{duration:02}:00')
                self.countdown_left_seconds = duration * 60
                break

    def show_video_image(self, pixmap):
        self.camera_label.setPixmap(pixmap)

    def obj_detected(self, detections: list):
        print(f'{len(detections)} object(s) detected.')
        hand_num = 0
        for detection, _, _ in detections:
            print(f'{detection=}')
            if detection.strip() == 'cell phone':
                self.killTimer(self.timer_id)
                self.video_worker.stop_detecting()
                QMessageBox.information(self, 'Attention', 
                'You are playing e-devices. The current session failed.',
                QMessageBox.Ok, QMessageBox.Ok)
                self.show_initial_lcd_coundown()
                self.action_button.setEnabled(True)
            elif detection.strip() == 'hand':
                hand_num = hand_num + 1
        print(f'{hand_num=}')

    def action_start(self):
        self.action_button.setEnabled(False)
        self.timer_id = self.startTimer(1000)
        self.video_worker.start_detecting()        

    def timerEvent(self, a0: 'QTimerEvent') -> None:
        self.countdown_left_seconds = self.countdown_left_seconds - 1
        min, sec = divmod(self.countdown_left_seconds, 60)
        self.countdown_lcd.display(f'{min:02}:{sec:02}')
        if self.countdown_left_seconds == 0:
            self.killTimer(self.timer_id)
            self.video_worker.stop_detecting()
            QMessageBox.information(self, 'Congratulations', 
            'You have finished this session successfully.',
            QMessageBox.Ok, QMessageBox.Ok)
            self.show_initial_lcd_coundown()
            self.action_button.setEnabled(True)

def detect(conn):
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    prog_config_file_path = os.path.join(cwd, 'config.ini')

    exception_weight_file_path = ''

    hand_model_cfg_file_path = ''
    hand_class_name_file_path = os.path.join(cwd, 'cfg/hand.names')
    hand_weight_file_path = ''
    
    if os.path.exists(prog_config_file_path):
        exception_config = get_prog_config(prog_config_file_path, 'yolo_exception_objects')
        exception_weight_file_path = exception_config.get('weight_file', exception_weight_file_path)

        hand_config = get_prog_config(prog_config_file_path, 'yolo_hands')
        hand_model_cfg_file_path = hand_config.get('model_cfg_file', hand_model_cfg_file_path)
        hand_class_name_file_path = hand_config.get('class_name_file', hand_class_name_file_path)
        hand_weight_file_path = hand_config.get('weight_file', hand_weight_file_path)

    # hack
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov3-spp.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    options = parser.parse_args(['--weights', exception_weight_file_path])
    exception_detector = yolov3_detect(options)
    exception_detector.send(None) # initialize

    hand_network, hand_class_names, hand_class_colors = darknet.load_network(
        hand_model_cfg_file_path,
        hand_class_name_file_path,
        hand_weight_file_path,
        batch_size=1
    )
    print(f'Loaded classes: {len(hand_class_names)}')

    hand_network_width = darknet.network_width(hand_network)
    hand_network_height = darknet.network_height(hand_network)
    print(f'Network: {hand_network_width=},{hand_network_height=}')

    while True:
        has_data = conn.poll(5)
        if not has_data:
            continue

        frames = conn.recv()
        if isinstance(frames, str) and frames == 'end':
            conn.close()
            break

        #print(f'Frame received at: {time.time()}')

        exception_detections = exception_detector.send(frames[0])

        frame_rgb = frames[1]
        frame_resized = cv.resize(frame_rgb, (hand_network_width, hand_network_height),
                                interpolation=cv.INTER_LINEAR)

        img_for_detect = darknet.make_image(hand_network_width, hand_network_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

        #print(f'Detecting hands at timestamp: {time.time()}')
        hand_detections = darknet.detect_image(hand_network, hand_class_names, img_for_detect, thresh=0.25)
        #print(f'Detecting of hands finished at timestamp: {time.time()}')
        darknet.print_detections(hand_detections)

        darknet.free_image(img_for_detect)

        conn.send(exception_detections + hand_detections)

class VideoWorker(QThread):
    image_ready_signal = pyqtSignal(QPixmap)
    detection_signal = pyqtSignal(list)

    def __init__(self, frequent = 16):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.mutex = QMutex()

        self.video_src = cv.VideoCapture(0)
        self.detecting = False

    def run(self):
        parent_conn, child_conn = mp.Pipe()
        detecting_process = mp.Process(target=detect, args=(child_conn,))
        detecting_process.start()

        frame_num = 0
        frame_processing = False
        while self.video_src.isOpened():
            #with QMutexLocker(self.mutex):
            if self.stopped:
                self.video_src.release()
                parent_conn.send('end')
                break
            
            succeeded, frame = self.video_src.read()
            if not succeeded: break

            if frame.ndim == 3:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            elif frame.ndim == 2:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
            else:
                frame_rgb = frame # not sure

            frame_rgb = cv.flip(frame_rgb, 1)
            height, width = frame_rgb.shape[:2]
            temp_image = QImage(frame_rgb.flatten(), width, height, QImage.Format_RGB888)
            temp_pixmap = QPixmap.fromImage(temp_image)
            self.image_ready_signal.emit(temp_pixmap)

            frame_num = frame_num + 1
            if frame_processing:
                has_data = parent_conn.poll()
                if has_data:
                    detections = parent_conn.recv()
                    frame_processing = False
                    if len(detections) and self.detecting:
                        self.detection_signal.emit(detections)
            elif self.detecting:
                parent_conn.send([frame, frame_rgb])
                frame_processing = True
        
        detecting_process.join()

    def stop(self):
        #with QMutexLocker(self.mutex):
        self.stopped = True

    def is_stopped(self):
        #with QMutexLocker(self.mutex):
        return self.stopped

    def start_detecting(self):
        #with QMutexLocker(self.mutex):
        self.detecting = True

    def stop_detecting(self):
        #with QMutexLocker(self.mutex):
        self.detecting = False

    def set_fps(self, fps):
        self.frequent = fps

if __name__=="__main__":
    app = QApplication(sys.argv)
    study = CountdownStudy()
    study.show()
    ret = app.exec_()
    sys.exit(ret)
