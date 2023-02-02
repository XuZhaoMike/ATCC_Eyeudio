import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.optimizers import RMSprop

from pynput.mouse import Button, Controller
from threading import Thread


class T_click():

    def __init__(self):
        self.num_tongue_frames = 2000
        self.num_other_frames = 4000
        self.face_cascade = cv2.CascadeClassifier(cv2.samples.findFile('click/haarcascade_frontalface_default.xml'))
        self.mouse = Controller()

        self.tongue_out = 0

        self.camstart = False
        self.frame = None

        t1 = Thread(target=self.t_click_detect_continuously, args=())
        t1.start()
    
    def get_data(self, img):
        tongue_out_examples = []
        other_examples = []

        print('stick the tip of your tongue out and scan the screen')
        for i in range(self.num_tongue_frames):
            # ret, img = vid.read()
            img = self.frame
            # cv2.imshow('frame', img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32), flags=cv2.CASCADE_SCALE_IMAGE)
            for (x,y,w,h) in faces:
                roi = gray[y:y + h, x:x+w]
                roi = cv2.resize(roi, (32, 32))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                tongue_out_examples.append(roi)
                print('time remaining: ', self.num_tongue_frames - i)


        print('great job! now relax your face to a normal resting position and scan the screen until the next message')
        for i in range(self.num_other_frames // 2):
            # ret, img = vid.read()
            img = self.frame
            # cv2.imshow('frame', img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            for (x,y,w,h) in faces:
                roi = gray[y:y + h, x:x+w]
                roi = cv2.resize(roi, (32, 32))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                other_examples.append(roi)
                print('time remaining: ', self.num_other_frames // 2 - i)

        print('almost done! now make various common facial expressions and scan the screen until prompted')
        for i in range(self.num_other_frames // 2):
            # ret, img = vid.read()
            img = self.frame
            # cv2.imshow('frame', img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            for (x,y,w,h) in faces:
                roi = gray[y:y + h, x:x+w]
                roi = cv2.resize(roi, (32, 32))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                other_examples.append(roi)
                print('time remaining: ', self.num_other_frames // 2 - i)

        print('data collection complete!')
        tongue_out_examples_arr = np.array(tongue_out_examples)
        tongue_out_examples_arr.shape
        tongue_out_examples_arr = tongue_out_examples_arr.reshape(tongue_out_examples_arr.shape[0], 32, 32)
        tongue_out_examples_arr.shape
        positives = np.ones(tongue_out_examples_arr.shape[0])
        other_examples_arr = np.array(other_examples)
        other_examples_arr.shape
        other_examples_arr = other_examples_arr.reshape(other_examples_arr.shape[0], 32, 32)
        negatives = np.zeros(other_examples_arr.shape[0])

        x = np.concatenate((tongue_out_examples_arr, other_examples_arr), axis = 0)
        y = np.concatenate((positives, negatives), axis = 0)
        indices = np.random.permutation(x.shape[0])
        n = int(x.shape[0]*0.8)
        training_idx, test_idx = indices[:n], indices[n:]
        x_train, x_test = x[training_idx,:], x[test_idx,:]
        y_train, y_test = y[training_idx], y[test_idx]

        return (x_train, y_train, x_test, y_test)

    def make_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(32, 32, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(.2),

            # tf.keras.layers.Conv2D(16, (4,4), activation='relu', input_shape=(32, 32, 1)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.MaxPooling2D(2, 2),
            # tf.keras.layers.Dropout(.2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(196, activation='relu'),
            tf.keras.layers.Dense(98, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def detect(self, model, img):
        # ret, img = vid.read()
        # cv2.imshow('frame', img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x,y,w,h) in faces:
            roi = gray[y:y + h, x:x+w]
            roi = cv2.resize(roi, (32, 32))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            tongue_out = model.predict(roi)[0]
            if tongue_out > 0.5:
                print('tongue out! = ', tongue_out)
                self.tongue_out = 1
            else:
                self.tongue_out = 0
            return tongue_out

    def t_click_train(self):
        # vid = cv2.VideoCapture(0)
        (x_train, y_train, x_test, y_test) = self.get_data(self.frame)

        model = self.make_model()
        model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])
        model.fit(x_train.reshape(x_train.shape[0], 32,32,1), y_train, epochs=10)
        model.evaluate(x_test.reshape(x_test.shape[0], 32,32,1), y_test)
        model.save('click/saved_t_click_model/')
        # vid.release()
        # cv2.destroyAllWindows()
        return model
    
    def t_click_detect_continuously(self):
        # vid = cv2.VideoCapture(0)
        model = tf.keras.models.load_model('click/saved_t_click_model/')

        last_five_list = [0,0,0,0,0]
        thresh = 0.6

        clicked_recent = False
        buffer = 5
        counter = 0

        while True:
            if not self.camstart:
                continue
            print("----------running tongue detecting---------")
            t_out = self.detect(model, self.frame)
            last_five_list.pop(0)
            last_five_list.append(t_out)

            avg = float(sum(filter(None, last_five_list))) / float(len(last_five_list))
            if avg > thresh:
                if clicked_recent:
                    counter += 1
                    if counter >= buffer:
                        clicked_recent = False
                else:
                    print('####### CLICK ACTIONED #######')
                    clicked_recent = True
                    self.mouse.press(Button.left)
                    self.mouse.release(Button.left)




if __name__ == '__main__':
    # T_click().t_click_train()
    T_click().t_click_detect_continuously()
    
