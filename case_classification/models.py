import tensorflow as tf
from tensorflow import keras
from keras import metrics,optimizers,losses
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Flatten,Add,BatchNormalization,Dense,Conv2D,MaxPooling2D,Dropout,GlobalAveragePooling2D,Input,concatenate

def make_predict(test_data):
    """
    Predicts target data using a previously trained model
    """

    loaded_model = keras.models.load_model("D:/BCDR/vgg19_trained",compile=True)
    
    with tf.device('/gpu'):
        predictions = loaded_model.predict(test_data)
    
    y_pred=[]
    for i in range(len(predictions)):
        y_pred.append(predictions[i][0])
    
    return y_pred

def load_model():
    try:
        loaded_model = keras.models.load_model("D:/BCDR/vgg19_trained",compile=True)
        print("Successfully loaded DL model!")
        return loaded_model
    except OSError:
        print("Failed to load DL model!") 
        return 

def create_sequential_model(size1,size2):

    """
    #Create Geras et al. model
    """

    model = Sequential()

    #first block
    model.add(Conv2D(32,kernel_size=(3, 3),strides=(2,2),input_shape=(size1,size2,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(3,3)))

    #second block
    model.add(Conv2D(64,kernel_size=(3, 3),strides=(2, 2),activation='relu'))
    model.add(Conv2D(64,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(Conv2D(64,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))

    #third block
    model.add(Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))

    #fourth block
    model.add(Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))

    #fifth block
    model.add(Conv2D(256,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(Conv2D(256,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(Conv2D(256,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(GlobalAveragePooling2D())

    #Fully Connected Layer
    model.add(Dense(1024,activation='relu'))
    #model.add(Dropout(0.2))

    #Output Layer
    model.add(Dense(2,activation='softmax'))

    return model

