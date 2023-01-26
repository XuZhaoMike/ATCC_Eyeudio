import argparse
import logging
import pathlib
import warnings

import torch
from omegaconf import DictConfig, OmegaConf

from .demo import Demo
from .utils import (check_path_all, download_dlib_pretrained_model,
                    download_ethxgaze_model, download_mpiifacegaze_model,
                    download_mpiigaze_model, expanduser_all,
                    generate_dummy_camera_params)
from threading import Thread, Lock
from time import sleep, ctime
import numpy as np
import cv2
import copy
import pyautogui


class Tracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lock = Lock()
        self.calib_start = False
        self.draw_next = False
        self.use_eye = False

        args = self.parse_args()
        t1 = Thread(target=self.work1, args=(args,))
        t1.start()

        t2 = Thread(target=self.work2, args=())
        t2.start()

        if self.use_eye:
            t3 = Thread(target=self.work3, args=())
            t3.start()

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--config',
            type=str,
            help='Config file. When using a config file, all the other '
            'commandline arguments are ignored. '
            'See https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/configs/eth-xgaze.yaml'
        )
        parser.add_argument(
            '--mode',
            type=str,
            default='mpiifacegaze',
            choices=['mpiigaze', 'mpiifacegaze', 'eth-xgaze'],
            help='With \'mpiigaze\', MPIIGaze model will be used. '
            'With \'mpiifacegaze\', MPIIFaceGaze model will be used. '
            'With \'eth-xgaze\', ETH-XGaze model will be used.')
        parser.add_argument(
            '--face-detector',
            type=str,
            default='mediapipe',
            choices=[
                'dlib', 'face_alignment_dlib', 'face_alignment_sfd', 'mediapipe'
            ],
            help='The method used to detect faces and find face landmarks '
            '(default: \'mediapipe\')')
        parser.add_argument('--device',
                            type=str,
                            choices=['cpu', 'cuda'],
                            help='Device used for model inference.')
        parser.add_argument('--image',
                            type=str,
                            help='Path to an input image file.')
        parser.add_argument('--video',
                            type=str,
                            help='Path to an input video file.')
        parser.add_argument(
            '--camera',
            type=str,
            help='Camera calibration file. '
            'See https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/calib/sample_params.yaml'
        )
        parser.add_argument(
            '--output-dir',
            '-o',
            type=str,
            help='If specified, the overlaid video will be saved to this directory.'
        )
        parser.add_argument('--ext',
                            '-e',
                            type=str,
                            choices=['avi', 'mp4'],
                            help='Output video file extension.')
        parser.add_argument(
            '--no-screen',
            action='store_true',
            help='If specified, the video is not displayed on screen, and saved '
            'to the output directory.')
        parser.add_argument('--debug', action='store_true')
        return parser.parse_args()


    def load_mode_config(self, args: argparse.Namespace) -> DictConfig:
        package_root = pathlib.Path(__file__).parent.resolve()
        if args.mode == 'mpiigaze':
            path = package_root / 'data/configs/mpiigaze.yaml'
        elif args.mode == 'mpiifacegaze':
            path = package_root / 'data/configs/mpiifacegaze.yaml'
        elif args.mode == 'eth-xgaze':
            path = package_root / 'data/configs/eth-xgaze.yaml'
        else:
            raise ValueError
        config = OmegaConf.load(path)
        config.PACKAGE_ROOT = package_root.as_posix()

        if args.face_detector:
            config.face_detector.mode = args.face_detector
        if args.device:
            config.device = args.device
        if config.device == 'cuda' and not torch.cuda.is_available():
            config.device = 'cpu'
            warnings.warn('Run on CPU because CUDA is not available.')
        if args.image and args.video:
            raise ValueError('Only one of --image or --video can be specified.')
        if args.image:
            config.demo.image_path = args.image
            config.demo.use_camera = False
        if args.video:
            config.demo.video_path = args.video
            config.demo.use_camera = False
        if args.camera:
            config.gaze_estimator.camera_params = args.camera
        elif args.image or args.video:
            config.gaze_estimator.use_dummy_camera_params = True
        if args.output_dir:
            config.demo.output_dir = args.output_dir
        if args.ext:
            config.demo.output_file_extension = args.ext
        if args.no_screen:
            config.demo.display_on_screen = False
            if not config.demo.output_dir:
                config.demo.output_dir = 'outputs'

        return config

    def work1(self,args):
        self.lock.acquire()
        if args.debug:
            logging.getLogger('ptgaze').setLevel(logging.DEBUG)

        if args.config:
            config = OmegaConf.load(args.config)
        elif args.mode:
            config = self.load_mode_config(args)
        else:
            raise ValueError(
                'You need to specify one of \'--mode\' or \'--config\'.')
        expanduser_all(config)
        if config.gaze_estimator.use_dummy_camera_params:
            generate_dummy_camera_params(config)

        OmegaConf.set_readonly(config, True)
        self.logger.info(OmegaConf.to_yaml(config))

        if config.face_detector.mode == 'dlib':
            download_dlib_pretrained_model()
        if args.mode:
            if config.mode == 'MPIIGaze':
                download_mpiigaze_model()
            elif config.mode == 'MPIIFaceGaze':
                download_mpiifacegaze_model()
            elif config.mode == 'ETH-XGaze':
                download_ethxgaze_model()

        check_path_all(config)
        self.demo = Demo(config)
        self.lock.release()
        self.demo.run()

    
    def work2(self):
        screenWidth, screenHeight = pyautogui.size() # Get the size of the primary monitor.
        currentMouseX, currentMouseY = pyautogui.position()
        sleep(1) # make sure work1 can get the self.lock
        self.lock.acquire()
        sleep(1)
        CALIBRATION_INTERVAL = 3 # change this interval
        CURSOR_INTERVAL = 0.1
        self.lock.release()

        # first four results is used to calibration
        x_right, x_left, y_up, y_down = 0, 0, 0, 0 
        # min     max     min     max

        iteration = -1
        
        # initialize recent x,y value and average value, and calibration variables
        x_recent, y_recent, x_ave, y_ave = [], [], 0 ,0
        global num_p_c, num_p_r
        num_p_r = 5
        num_p_c = 4
        calib_p = np.zeros((num_p_c,num_p_r,2))
        
        calib_done = False

        while True:
            if self.use_eye:
                if len(self.demo.gaze_estimator.results)==0:
                    continue
                res = self.demo.gaze_estimator.results
                gazevec = np.average(res,axis=0)
                x,y,z = -gazevec
                y = -y
                face_x, face_y, face_z = self.demo.gaze_estimator.facecenter
                face_pitch =np.mean(self.demo.gaze_estimator.facerot[:,0])

                # Calibration - more point method
                if iteration == -1:
                    self.logger.info("-- Start Calibration, look at red dot (first will appear yop-left corner) --")
                    sleep(CALIBRATION_INTERVAL)
                    self.calib_start = True
                    self.logger.info("------------------- Look at first location -------------------")
                    sleep(CALIBRATION_INTERVAL)
                    iteration+=1
                    continue
                if iteration < num_p_r*num_p_c:
                    row = int(iteration/num_p_r)
                    col = (iteration%num_p_r)
                    calib_p[row,col,0]=x
                    calib_p[row,col,1]=y
                    if iteration == num_p_r*num_p_c-1:
                        calib_distance = face_z
                        calib_facepitch = face_pitch
                        self.logger.info("-------------------------------------- Finished --------------------------------------")
                    else:
                        self.logger.info("------------------- Then Look at next location -------------------")
                    self.draw_next = True
                    sleep(CALIBRATION_INTERVAL)
                    iteration += 1
                    continue
                if not calib_done:
                    self.logger.info("\nBefore calibration: \n calib_p{}".format(calib_p))
                    for j in range(num_p_r):
                        temp_x = np.average(calib_p[:,j,0])
                        calib_p[:,j,0] = temp_x
                    for i in range(num_p_c):
                        temp_y = np.average(calib_p[i,:,1])
                        calib_p[i,:,1] = temp_y
                    calib_done = True
                    self.logger.info("\nFinished calibration: \n calib_p{}".format(calib_p))
                
                # adjust y based on face pitch
                self.logger.info("\n unadjusted----- x:{}   y: {}  z: {}".format(x, y, z))
                pitch = (face_pitch - calib_facepitch)/100
                pitch = 0
                g_y = np.matrix([[1, 0, 0],
                        [0, np.cos(pitch), -np.sin(pitch)],
                        [0, np.sin(pitch), np.cos(pitch)]])
                old_p = np.matrix([[x],[y],[z]])
                new_p = g_y @ old_p
                x,y,z = new_p[0,0], new_p[1,0], new_p[2,0]
                
                # adjust x,y based on face center
                # x = x * face_z / calib_distance - face_x
                # y = y * face_z / calib_distance             #TODO: find ways to properly adjust y
                self.logger.info("\n unsacled----- x:{}   y: {}  z: {}, pitch:{}".format(x, y,z, pitch))

                # grid way to polyfit scale x and y
                if x<calib_p[0,2,0]:
                    screen_x = [0,screenWidth/4-1,screenWidth/2-1]
                    ploy_x = np.polyfit(calib_p[0,0:3,0],screen_x,2)
                else:
                    screen_x = [screenWidth/2-1,screenWidth*3/4-1,screenWidth-1]
                    ploy_x = np.polyfit(calib_p[0,2:5,0],screen_x,2)
                px = np.poly1d(ploy_x)
                x = px(x)
                # screen_y = [0,screenHeight/2-1,screenHeight-1] # 3 calib point
                screen_y = [0,screenHeight/3-1,screenHeight*2/3-1,screenHeight-1] # 4 calib point
                ploy_y = np.polyfit(calib_p[:,0,1],screen_y,2)
                py = np.poly1d(ploy_y)
                y = py(y)
                # yp = np.linspace(0.05,0.25,100)
                # plt.plot(yp,py(yp))
                # plt.show()

                # grid way to linearly scale x and y
                # for j in range(0,num_p_r):
                #     if j==num_p_r-1:
                #         x = screenWidth
                #     elif x < calib_p[0,0,0]:
                #         x = 0
                #         break
                #     elif x > calib_p[0,j,0] and x < calib_p[0,j+1,0]:
                #         x = ((x - calib_p[0,j,0]) / (calib_p[0,j+1,0] - calib_p[0,j,0]) + j) * (screenWidth/(num_p_r-1))
                #         break
                # for i in range(0,num_p_c):
                #     if i==num_p_c-1:
                #         y = screenHeight
                #     elif y < calib_p[0,0,1]:
                #         y = 0
                #         break
                #     elif y > calib_p[i,0,1] and y < calib_p[i+1,0,1]:
                #         y = ((y - calib_p[i,0,1]) / (calib_p[i+1,0,1] - calib_p[i,0,1]) + i) * (screenHeight/(num_p_c-1))
                #         break

                # scale x and y
                # x = (x - x_left) / (x_right - x_left) * (screenWidth)
                # y = (y - y_up) / (y_down - y_up) * (screenHeight)
                # self.logger.info("\n x:{}   y: {}".format(x, y))

                # bound check
                if x <= 0:
                    x = 1
                if x >= screenWidth:
                    x = screenWidth - 2
                if y <= 0:
                    y = 1
                if y >= screenHeight:
                    y = screenHeight - 2

                # store recent x,y value and average value
                k = 0.15
                if len(x_recent)<10:
                    x_recent.append(x)
                    y_recent.append(y)
                    x_ave = sum(x_recent)/len(x_recent)
                    y_ave = sum(y_recent)/len(y_recent)
                else:
                    if abs(x-x_ave)<150 and abs(y-y_ave)<150:
                        self.logger.info("----------------------using average------------------")
                        pyautogui.moveTo(x_ave+(x-x_ave)*k, y_ave+(y-y_ave)*k)
                        sleep(CURSOR_INTERVAL)
                        continue
                    else:
                        x_recent = []
                        y_recent = []
                pyautogui.moveTo(x, y) # x, y  positive number

                sleep(CURSOR_INTERVAL)
                iteration += 1
            else:
                face_x, face_y, face_z = self.demo.gaze_estimator.facecenter
                temp_rot = []
                for r in self.demo.gaze_estimator.facerot:
                    temp_rot.append(r)
                face_pitch, face_yaw, face_roll =np.average(temp_rot,axis=0)
                self.logger.info("\n face_x:{} face_y: {} face_z:{}".format(face_x, face_y, face_z))
                self.logger.info("\n pitch:{}   yaw: {}   roll:{}".format(face_pitch, face_yaw, face_roll))
                if iteration==-1:
                    self.logger.info("-- Start Calibration, getting default head orientation --")
                    face_pitch_d = face_pitch
                    face_yaw_d = face_yaw
                    face_x_d = face_x
                    face_y_d = face_y
                    iteration+=1
                else:
                    # linearly scale x and y
                    x = (face_yaw-face_yaw_d)*(screenWidth/50) - (face_x-face_x_d)*(screenWidth/20) + screenWidth/2
                    y = -(face_pitch-face_pitch_d)*(screenHeight/20) + (face_y-face_y_d)*(screenHeight/20) + screenHeight/2
                    # bound check
                    if x <= 0:
                        x = 1
                    if x >= screenWidth:
                        x = screenWidth - 2
                    if y <= 0:
                        y = 1
                    if y >= screenHeight:
                        y = screenHeight - 2
                    # smoothing by storing recent x,y value and average value
                    k = 0.15
                    if len(x_recent)<10:
                        x_recent.append(x)
                        y_recent.append(y)
                    else:
                        x_recent.pop(0)
                        y_recent.pop(0)
                        x_recent.append(x)
                        y_recent.append(y)
                    x_ave = sum(x_recent)/len(x_recent)
                    y_ave = sum(y_recent)/len(y_recent)
                    if (x-x_ave)**2+(y-y_ave)**2<22500:
                        self.logger.info("----------------------using average------------------")
                        pyautogui.moveTo(x_ave+(x-x_ave)*k, y_ave+(y-y_ave)*k)
                        sleep(CURSOR_INTERVAL)
                        continue
                    else:
                        x_recent = []
                        y_recent = []
                    # move cursor
                    pyautogui.moveTo(x, y) # x, y  positive number
                    sleep(CURSOR_INTERVAL)
                


    def work3(self):
        draw_r = 10
        target_img = cv2.imread('white.jpg')
        drawn_img = copy.deepcopy(target_img)
        width = int(target_img.shape[1])
        height = int(target_img.shape[0])
        while True:
            if self.calib_start:
                break
        sleep(0.03)
        cv2.namedWindow('target',cv2.WND_PROP_FULLSCREEN)        # Create a named window
        cv2.imshow('target',drawn_img)
        cv2.setWindowProperty('target', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        for i in range(num_p_r*num_p_c):
            img_row = int(i/num_p_r)
            img_col = (i%num_p_r)
            img_x = int(img_col/(num_p_r-1)*width-1)
            img_y = int(img_row/(num_p_c-1)*height-1)
            cv2.circle(drawn_img,(img_x,img_y),draw_r,(0,0,255),-1)
            while True:
                if self.draw_next:
                    self.draw_next = False
                    break
                cv2.imshow('target',drawn_img)
                cv2.waitKey(1)
            drawn_img = copy.deepcopy(target_img)
        cv2.destroyWindow('target')

if __name__ =='__main__':
    Tracker()
