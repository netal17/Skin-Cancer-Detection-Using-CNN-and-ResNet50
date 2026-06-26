
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

IMG_SIZE=(224,224)
BATCH_SIZE=32
EPOCHS=20

train_dir="dataset/train"
test_dir="dataset/test"

train_gen=ImageDataGenerator(rescale=1./255,validation_split=0.2)
test_gen=ImageDataGenerator(rescale=1./255)

train_data=train_gen.flow_from_directory(
    train_dir,target_size=IMG_SIZE,batch_size=BATCH_SIZE,
    class_mode="categorical",subset="training")

val_data=train_gen.flow_from_directory(
    train_dir,target_size=IMG_SIZE,batch_size=BATCH_SIZE,
    class_mode="categorical",subset="validation")

test_data=test_gen.flow_from_directory(
    test_dir,target_size=IMG_SIZE,batch_size=BATCH_SIZE,
    class_mode="categorical",shuffle=False)

def build_cnn():
    model=Sequential([
        Conv2D(64,(3,3),activation="relu",input_shape=(224,224,3)),
        MaxPooling2D(),
        Dropout(0.25),
        Conv2D(64,(3,3),activation="relu"),
        MaxPooling2D(),
        Dropout(0.25),
        Flatten(),
        Dense(128,activation="relu"),
        Dense(2,activation="softmax")
    ])
    model.compile(optimizer=Adam(1e-4),loss="categorical_crossentropy",metrics=["accuracy"])
    return model

cnn=build_cnn()
lr=ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3)

cnn.fit(train_data,validation_data=val_data,epochs=EPOCHS,callbacks=[lr])
cnn.save("cnn_skin_cancer.keras")

base=ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3),pooling="avg")
for l in base.layers:
    l.trainable=False

from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

x=Dense(2,activation="softmax")(base.output)
resnet=Model(inputs=base.input,outputs=x)
resnet.compile(optimizer=Adam(1e-4),loss="categorical_crossentropy",metrics=["accuracy"])
resnet.fit(train_data,validation_data=val_data,epochs=EPOCHS,callbacks=[lr])
resnet.save("resnet50_skin_cancer.keras")

print("CNN:",cnn.evaluate(test_data))
print("ResNet50:",resnet.evaluate(test_data))
