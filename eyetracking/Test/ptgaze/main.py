import argparse
import logging
import pathlib
import warnings

import torch
from omegaconf import DictConfig, OmegaConf

from demo import Demo
from utils import (check_path_all, download_dlib_pretrained_model,
                    download_ethxgaze_model, download_mpiifacegaze_model,
                    download_mpiigaze_model, expanduser_all,
                    generate_dummy_camera_params)
from threading import Thread, Lock
from time import sleep, ctime
import numpy as np
import cv2
import copy

logger = logging.getLogger(__name__)
lock = Lock()
calib_start = False
draw_next = False

def parse_args() -> argparse.Namespace:
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


def load_mode_config(args: argparse.Namespace) -> DictConfig:
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


def main():
    args = parse_args()
    t1 = Thread(target=work1, args=(args,))
    t1.start()

    t2 = Thread(target=work2, args=())
    t2.start()


def work1(args):
    global demo
    lock.acquire()
    if args.debug:
        logging.getLogger('ptgaze').setLevel(logging.DEBUG)

    if args.config:
        config = OmegaConf.load(args.config)
    elif args.mode:
        config = load_mode_config(args)
    else:
        raise ValueError(
            'You need to specify one of \'--mode\' or \'--config\'.')
    expanduser_all(config)
    if config.gaze_estimator.use_dummy_camera_params:
        generate_dummy_camera_params(config)

    OmegaConf.set_readonly(config, True)
    logger.info(OmegaConf.to_yaml(config))

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
    demo = Demo(config)
    lock.release()
    demo.run()

import pyautogui
def work2():
    screenWidth, screenHeight = pyautogui.size() # Get the size of the primary monitor.
    currentMouseX, currentMouseY = pyautogui.position()
    sleep(1) # make sure work1 can get the lock
    lock.acquire()
    sleep(1)
    global demo
    CALIBRATION_INTERVAL = 3 # change this interval
    CURSOR_INTERVAL = 0.1
    lock.release()

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
        x = 0
        y = 0
        for res in demo.gaze_estimator.results:
            x += res[0]
            y += res[1]
        
        # If the demo gaze result has 0 as an initial value, uncomment the +0.01
        if len(demo.gaze_estimator.results)!=0:
            x /= -len(demo.gaze_estimator.results)
            y /= len(demo.gaze_estimator.results) 
        face_x, face_y, face_z = demo.gaze_estimator.facecenter
        
        # calibration
        # if iteration == 0:
        #     logger.info("------------------- Look Upper-left -------------------")
        #     sleep(CALIBRATION_INTERVAL)
        #     iteration += 1
        #     continue
        # if iteration == 1: # upper-left
        #     x_left += x
        #     y_up += y
        #     logger.info("------------------- Then Look Upper-right -------------------")
        #     sleep(CALIBRATION_INTERVAL)
        #     iteration += 1
        #     continue
        # elif iteration == 2: # upper-right
        #     x_right += x
        #     y_up += y
        #     logger.info("------------------- Then Look lower-right -------------------")
        #     sleep(CALIBRATION_INTERVAL)
        #     iteration += 1
        #     continue
        # elif iteration == 3: # lower-right
        #     x_right += x
        #     y_down += y
        #     logger.info("------------------- Then Look lower-left -------------------")
        #     sleep(CALIBRATION_INTERVAL)
        #     iteration += 1
        #     continue
        # elif iteration == 4: # lower-left
        #     x_left += x
        #     y_down += y
        #     calib_distance = face_z
        #     logger.info("-------------------------------------- Finished --------------------------------------")
        #     sleep(CALIBRATION_INTERVAL)
        #     iteration += 1
        #     continue
        # elif iteration == 5:
        #     x_right, x_left, y_up, y_down = x_right / 2, x_left / 2, y_up / 2, y_down / 2
        #     logger.info("\nFinished calibration: \n x_right {}, \n x_left {}, \n y_up {}, \n y_down {}".format(x_right, x_left, y_up, y_down))

        # Calibration - more point method
        if iteration == -1:
            logger.info("-- Start Calibration, look at red dot (first will appear yop-left corner) --")
            sleep(CALIBRATION_INTERVAL)
            global calib_start
            calib_start = True
            logger.info("------------------- Look at first location -------------------")
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
                logger.info("-------------------------------------- Finished --------------------------------------")
            else:
                logger.info("------------------- Then Look at next location -------------------")
            global draw_next
            draw_next = True
            sleep(CALIBRATION_INTERVAL)
            iteration += 1
            continue
        if not calib_done:
            logger.info("\nBefore calibration: \n calib_p{}".format(calib_p))
            for j in range(num_p_r):
                temp_x = np.average(calib_p[:,j,0])
                calib_p[:,j,0] = temp_x
            for i in range(num_p_c):
                temp_y = np.average(calib_p[i,:,1])
                calib_p[i,:,1] = temp_y
            calib_done = True
            logger.info("\nFinished calibration: \n calib_p{}".format(calib_p))
        
        # adjust x,y based on face center
        x = x * face_z / calib_distance - face_x
        y = y * face_z / calib_distance             #TODO: find ways to properly adjust y
        logger.info("\n unsacled----- x:{}   y: {}, face_x:{}".format(x, y,face_x))

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
        # logger.info("\n x:{}   y: {}".format(x, y))

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
                logger.info("----------------------using average------------------")
                pyautogui.moveTo(x_ave+(x-x_ave)*k, y_ave+(y-y_ave)*k)
                sleep(CURSOR_INTERVAL)
                continue
            else:
                x_recent = []
                y_recent = []
        pyautogui.moveTo(x, y) # x, y  positive number

        sleep(CURSOR_INTERVAL)
        iteration += 1

def work3():
    global draw_next
    draw_r = 10
    target_img = cv2.imread('white.jpg')
    drawn_img = copy.deepcopy(target_img)
    width = int(target_img.shape[1])
    height = int(target_img.shape[0])
    while True:
        if calib_start:
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
            if draw_next:
                draw_next = False
                break
            cv2.imshow('target',drawn_img)
            cv2.waitKey(1)
        drawn_img = copy.deepcopy(target_img)
    cv2.destroyWindow('target')

args = parse_args()
t1 = Thread(target=work1, args=(args,))
t1.start()

t2 = Thread(target=work2, args=())
t2.start()

t3 = Thread(target=work3, args=())
t3.start()