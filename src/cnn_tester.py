
import dataset_builder as db
import skin_reco
import numpy as np
import cv2
import os
import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array

CATEGORIES = ["A","AA","U","w","y"]

def max_index_of(array):
    m = -1
    index = -1
    for i in range(len(array)):
        if array[i] > m:
            m = array[i]
            index = i
    return index

def main():
    # load neural network
    #model = cnn.read_model('../model')
    model = tf.keras.models.load_model("K:/python/Cam/src/sign_language_sinhala.model")


    # load face reco haar
    face_cascade = cv2.CascadeClassifier('../haar/haarcascade_frontalface_default.xml')

    # init camera
    cap = cv2.VideoCapture(0)

    while(True):
        # get image from camera
        ret, frame = cap.read()
        cv2.rectangle(frame, (100,100), (300,300), (0,0,255), 1)
        #area = frame[100:300, 100:300]

        # extract hand using skin color
        lower_range, upper_range = skin_reco.hsv_color_range_from_image(frame, face_cascade)
        if lower_range is not None and upper_range is not None:
            result = skin_reco.filter_skin(frame[100:300, 100:300], lower_range, upper_range)

            # suit the image for the network: reshape, normalize
            image = cv2.resize(result, (db.width, db.height))
            image = img_to_array(image)
            image = np.array(image, dtype="float") / 255.0
            image = image.reshape(1, db.width, db.height, db.channel)
            #image = image.reshape(1, 50, 50, 1)

            # use the model to predict the output
            output = model.predict(image)
            #prediction = model.predict([prepare('K:/Bsc_ICT/FinalProject/Codes/Simple-Sign-Language-Detector-master/mydata/test_set/D/1.png')])

            pred_name = CATEGORIES[np.argmax(output)]
            key = cv2.waitKey(1)
            if key == ord('x'):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,pred_name, (0,130), font, 1, (200,255,255), 2, cv2.LINE_AA)

                
            

            print(pred_name)

            cv2.imshow('result', result)
        

        

        # display
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
