import tensorflow as tf
from tensorflow import keras
from keras import metrics,callbacks,layers
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D,Dropout,GlobalAveragePooling2D,Input, concatenate

def train_model(model,optimizer,loss_function,metrics,epochs,steps_per_epoch,training_data,callbacks):
    """
    Fits the model to target data using said hyperparameters
    """

    model.compile(
        optimizer= str(optimizer),
        loss= loss_function,
        metrics= metrics)
    
    with tf.device('/gpu'):
        hist = model.fit(training_data,
        epochs= epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks = callbacks
        )
    
    return hist

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

def create_functional_model(size1,size2):
    """
    Create multiple input model
    """
#Craniocaudal Block
    #Input
    cc_image_input = keras.Input(shape=(size1,size2,1),name='CC_Input')
    cc_image_1 = keras.layers.Conv2D(32,kernel_size=(3, 3),strides=(2,2),activation='relu')(cc_image_input)
    cc_image_2 = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(3,3))(cc_image_1)
    #First Block
    cc_image_3 = keras.layers.Conv2D(64,kernel_size=(3, 3),strides=(2, 2),activation='relu')(cc_image_2)
    cc_image_4 = keras.layers.Conv2D(64,kernel_size=(3, 3),strides=(1, 1),activation='relu')(cc_image_3)
    cc_image_5 = keras.layers.Conv2D(64,kernel_size=(3, 3),strides=(1, 1),activation='relu')(cc_image_4)
    cc_image_6 = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2, 2))(cc_image_5)
    #Second Block
    cc_image_7 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(cc_image_6)
    cc_image_8 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(cc_image_7)
    cc_image_9 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(cc_image_8)
    cc_image_10 = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2, 2))(cc_image_9)
    #Third Block
    cc_image_11 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(cc_image_10)
    cc_image_12 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(cc_image_11)
    cc_image_13 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(cc_image_12)
    cc_image_14 = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2, 2))(cc_image_13)
    #Fourth Block   
    cc_image_15 = keras.layers.Conv2D(256,kernel_size=(3, 3),strides=(1, 1),activation='relu')(cc_image_14)
    cc_image_16 = keras.layers.Conv2D(256,kernel_size=(3, 3),strides=(1, 1),activation='relu')(cc_image_15)
    cc_image_17 = keras.layers.Conv2D(256,kernel_size=(3, 3),strides=(1, 1),activation='relu')(cc_image_16)
    cc_image_18 = keras.layers.GlobalAveragePooling2D()(cc_image_17)

#Mediolateral Oblique Block
    #Input
    mlo_image_input = keras.Input(shape=(size1,size2,1),name='MLO_Input')
    mlo_image_1 = keras.layers.Conv2D(32,kernel_size=(3, 3),strides=(2,2),activation='relu')(mlo_image_input)
    mlo_image_2 = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(3,3))(mlo_image_1)
    #First Block
    mlo_image_3 = keras.layers.Conv2D(64,kernel_size=(3, 3),strides=(2, 2),activation='relu')(mlo_image_2)
    mlo_image_4 = keras.layers.Conv2D(64,kernel_size=(3, 3),strides=(1, 1),activation='relu')(mlo_image_3)
    mlo_image_5 = keras.layers.Conv2D(64,kernel_size=(3, 3),strides=(1, 1),activation='relu')(mlo_image_4)
    mlo_image_6 = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2, 2))(mlo_image_5)
    #Second Block
    mlo_image_7 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(mlo_image_6)
    mlo_image_8 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(mlo_image_7)
    mlo_image_9 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(mlo_image_8)
    mlo_image_10 = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2, 2))(mlo_image_9)
    #Third Block
    mlo_image_11 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(mlo_image_10)
    mlo_image_12 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(mlo_image_11)
    mlo_image_13 = keras.layers.Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu')(mlo_image_12)
    mlo_image_14 = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2, 2))(mlo_image_13)
    #Fourth Block 
    mlo_image_15 = keras.layers.Conv2D(256,kernel_size=(3, 3),strides=(1, 1),activation='relu')(mlo_image_14)
    mlo_image_16 = keras.layers.Conv2D(256,kernel_size=(3, 3),strides=(1, 1),activation='relu')(mlo_image_15)
    mlo_image_17 = keras.layers.Conv2D(256,kernel_size=(3, 3),strides=(1, 1),activation='relu')(mlo_image_16)
    mlo_image_18 = keras.layers.GlobalAveragePooling2D()(mlo_image_17)

#Concatenation
    image_1 = keras.layers.concatenate([cc_image_18,mlo_image_18])
    image_2 = keras.layers.Dense(1024,activation='relu')(image_1)
    cc_output = keras.layers.Dense(2,activation='softmax')(image_2)
    mlo_output = keras.layers.Dense(2,activation='softmax')(image_2)

#Building the Model
    model = keras.Model(inputs=[cc_image_input,mlo_image_input],outputs=[cc_output,mlo_output],name = 'functional_model')

    return model

