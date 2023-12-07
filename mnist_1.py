import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf 
from tensorflow import keras
from keras.datasets import mnist
from keras import layers

#loads the mnist data set
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#Flattens out the images to a 1d array instead of a 2d one
x_train = x_train.reshape(-1,28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1,28*28).astype("float32") / 255.0


#Sequential API\ (Very convenient,not flexible)
#Makes the nerual network as a "shell"
model = keras.Sequential([
     
    keras.Input(shape=(28*28)),
    layers.Dense(512,activation="relu"),
    layers.Dense(256,activation="relu"),
    layers.Dense(10),
])


#You can do all of it individual used a lot for debugging
model = keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(256,activation="relu",name = 'second_layer'))
model.add(layers.Dense(10))


#Functional API (A bit more flexible)
inputs = keras.Input(shape=(28*28),name='inputs')
x = layers.Dense(512,activation='relu',name = 'first_layer')(inputs)#<-- uses previous layer as this ones input
x = layers.Dense(256,activation='relu',name = 'second_layer')(x)#<-- uses previous layer as this ones input
x = layers.Dense(128,activation='relu',name = 'third_layer')(x)#<-- uses previous layer as this ones input
outputs = layers.Dense(10,activation='softmax',name = "output")(x)#<-- uses previous layer as this ones input
model = keras.Model(inputs=inputs,outputs=outputs)


#The nerual networks "settings"
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)


#Trains the model
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=1)

model.evaluate(x_test,y_test,batch_size=32,verbose=1)
        