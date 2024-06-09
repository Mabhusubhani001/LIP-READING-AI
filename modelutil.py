#we'll use 3D conv layers to pass videos and condense it down to a classification dense layer which predicts char
#we'll use a special loss func called CTC loss func
#we'll use this loss func when we don't have word transriptions that aren't aligned to frames - it reduces the dupicates
from keras.models import Sequential
from keras.layers import Conv3D,Dense,LSTM,MaxPool3D,Flatten,Dropout,Bidirectional,Activation,Reshape,SpatialDropout3D,TimeDistributed,Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
Orthogonal=tf.keras.initializers.Orthogonal()
def load_model() -> Sequential:
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer=Orthogonal, return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer=Orthogonal, return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(os.path.join('models','checkpoints','model.h5'))

    return model