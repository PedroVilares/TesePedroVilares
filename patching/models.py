import tensorflow as tf
from tensorflow import keras
from keras import metrics,optimizers,losses
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Flatten,Add,BatchNormalization,Dense,Conv2D,MaxPooling2D,Dropout,GlobalAveragePooling2D,Input,concatenate

def train_model(model,optimizer,loss_function,metrics,epochs,steps_per_epoch,training_data,validation_data,callbacks):
    """
    Fits the model to target data using said hyperparameters
    """

    model.compile(
        optimizer= optimizer,
        loss= loss_function,
        metrics= metrics)
    
    with tf.device('/gpu'):
        hist = model.fit(training_data,
        epochs= epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data,
        callbacks = callbacks
        )
    
    return hist

def train_transferred_model(model,optimizer,loss_function,metrics,epochs,steps_per_epoch,training_data,validation_data,callbacks):

    model.compile(
        optimizer= optimizer,
        loss= loss_function,
        metrics= metrics)
    
    with tf.device('/gpu'):
        hist = model.fit(training_data,
        epochs= epochs,
        callbacks = callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_data = validation_data
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

def create_resnet_model(size1,size2):
    """
    Create ResNet model
    """

    image_input = keras.layers.Input(shape=(size1,size2,1))
    conv_1 = keras.layers.Conv2D(16, kernel_size=(7,7),strides=(2,2),activation='relu')(image_input)
    pool_1 = keras.layers.AveragePooling2D(pool_size=(3,3),strides=2)(conv_1)

    #ResNet Layers
    block_1 = resnet_block(input= pool_1,filter= 16,downsample= 1)
    block_2 = resnet_block(input= block_1,filter= 16,downsample= 1)
    block_3 = resnet_block(input= block_2,filter= 32,downsample= 2)
    block_4 = resnet_block(input= block_3,filter= 32,downsample= 1)
    block_5 = resnet_block(input= block_4,filter= 64,downsample= 2)
    block_6 = resnet_block(input= block_5,filter= 64,downsample= 1)
    block_7 = resnet_block(input= block_6,filter= 128,downsample= 2)
    block_8 = resnet_block(input= block_7,filter= 128,downsample= 1)
    block_9 = resnet_block(input= block_8,filter= 256,downsample= 2)
    block_10 = resnet_block(input= block_9,filter= 256,downsample= 1)

    batch_norm_1 = keras.layers.BatchNormalization()(block_10)
    activation_1 = keras.layers.Activation('relu')(batch_norm_1)

    #Fully-Connected Layers
    dense_1 = keras.layers.Dense(300,activation='relu')(activation_1)
    output_1 = keras.layers.Dense(2,activation='softmax')(dense_1)

    model = keras.models.Model(inputs = image_input,outputs=output_1, name = 'ResNet 22 model')

    return model

def resnet_block(input,filter,downsample):
    X = keras.layers.BatchNormalization()(input)
    X = keras.layers.Activation('relu')(X)
    if downsample != 1:
        X_shortcut = X
        X_shortcut = keras.layers.Conv2D(filters=filter,kernel_size=(1,1),strides=(downsample,downsample))(X_shortcut)
    X = keras.layers.Conv2D(filters= filter,kernel_size=(3,3),strides=(downsample,downsample))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Conv2D(filters= filter,kernel_size = (3,3))(X)
    if downsample != 1:
        X = keras.layers.Add()([X,X_shortcut])
    
    return X

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
    model = keras.models.Model(inputs=[cc_image_input,mlo_image_input],outputs=[cc_output,mlo_output],name = 'functional_model')

    return model

def create_patch_model(size1,size2):

    model = Sequential()

    #first block
    model.add(Conv2D(16,kernel_size=(3, 3),strides=(1,1),input_shape=(size1,size2,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #second block
    model.add(Conv2D(32,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #third block
    model.add(Conv2D(64,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #fourth block
    model.add(Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu'))

    #fifth block
    model.add(Conv2D(128,kernel_size=(3, 3),strides=(1, 1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())

    #dense one
    model.add(Dense(300,activation='relu'))
    model.add(Dropout(0.25))

    #dense two
    model.add(Dense(300,activation='relu'))
    #model.add(Dropout(0.3))

    #output
    model.add(Dense(1,activation='sigmoid'))

    return model
