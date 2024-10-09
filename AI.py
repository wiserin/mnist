from keras.api.models import load_model
from PIL import Image
import numpy as np

model = load_model('mnist_model.h5')

image = Image.open('image.png')

image = image.resize((28, 28))

image = image.convert('L')

file = np.array(image)

data = file.astype('float32') / 255.0
data = data.reshape(1, 28, 28, 1)

prediction = model.predict(data)

class_ = prediction.argmax()

print(class_)