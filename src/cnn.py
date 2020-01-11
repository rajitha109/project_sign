import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os
import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
import pickle
import time



DATADIR = "K:\python\Cam\sinhala_gestures"

    

CATEGORIES = ["A","AA","U","w","y"]





IMG_SIZE = 64
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


# In[4]:


training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        
        for img in os.listdir(path):
                    try:
                        img_array =cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        training_data.append([new_array, class_num])
                        
            
                    except Exception as e:
                        pass

            
        
        
        
create_training_data()
print(len(training_data))


random.shuffle(training_data)


X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)


for sample in training_data[:20]:
    print(sample[1])




X = X/255.0



dense_layers = [1]
layer_sizes = [32]
conv_layers = [3]


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
                NAME ="{}-conv-{}-nodes-{}-dense-{}-sign2".format(conv_layer, layer_size, dense_layer, int(time.time()))
                #tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
                print(NAME)

                model = Sequential()
                
                model.add( Conv2D(layer_size, (3,3), input_shape =X.shape[1:]) )
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
                
                for l in range(conv_layer-1): 

                    model.add(Conv2D(layer_size, (3,3)) )
                    model.add(Activation("relu"))
                    model.add(MaxPooling2D(pool_size=(2,2)))
                
                model.add(Flatten())
                    
                for l in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation("relu"))
                    
                    
            
               
             

                #model.add(Dropout(0.4))
                model.add(Dense(5))
                model.add(Activation('sigmoid'))

            
                
                model.compile(optimizer = optimizers.SGD(0.001),
                              loss = 'sparse_categorical_crossentropy',
                              metrics = ['accuracy'])


                history = model.fit(x_train, y_train, batch_size=32, epochs=20,validation_split=0.3)#, ''callbacks=[tensorboard]''

#accuracy Training

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



model.save_weights('final_weight.h5')
model.save("sign_language.h5")
