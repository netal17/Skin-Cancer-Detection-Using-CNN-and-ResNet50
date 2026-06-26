
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

model=load_model("resnet50_skin_cancer.keras")
img=image.load_img(sys.argv[1],target_size=(224,224))
x=image.img_to_array(img)/255.0
x=np.expand_dims(x,0)
pred=model.predict(x)[0]
classes=["Benign","Malignant"]
print(classes[np.argmax(pred)], pred)
