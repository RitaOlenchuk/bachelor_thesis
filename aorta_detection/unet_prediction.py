import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from keras.models import Model
from keras import layers as klayers
from keras.optimizers import Adam
from keras import backend as K
from keras.models import model_from_json

def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# load json and create model
json_file = open('/usr/local/hdd/rita/DL/model4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/usr/local/hdd/rita/DL/model4.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
model.compile(optimizer=Adam(lr=1e-5), loss=dice_coefficient_loss, metrics=[dice_coefficient, 'accuracy'])
print("After loading model")

input_shape = (960, 1280, 1)
#input_shape = (480, 640, 1)
test_path = "/usr/local/hdd/tfunet/aortack/sequences/test"
n_test_image = 10
endName = "small.tif"

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# load test images
test_image = np.empty(n_test_image * input_shape[0] * input_shape[1] )
test_image = test_image.reshape((n_test_image, ) + input_shape)

files = sorted([f for f in os.listdir(test_path) if f.endswith(endName)])
files = files[:n_test_image]

count = 0
for i in files:
    img_path = os.path.join(test_path, str(i))
    img = Image.open(img_path)
    img = np.asarray(img, dtype=np.float32)
    img = rgb2gray(img)
    img /= 255
    test_image[count, :, :, 0] = img[:input_shape[0], :input_shape[1]]
    count += 1

test_pred = model.predict(test_image)
test_pred = (test_pred > 0.5).astype(np.int)



def plot_test_pred(test_image, test_pred, testIdx=0):
    fig = plt.figure(figsize=(12, 12))

    plt.subplot(221)
    plt.imshow(test_image[testIdx, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.title('Test ' + str(testIdx), fontsize=16)

    plt.subplot(222)
    plt.imshow(test_pred[testIdx, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.title('Predict ' + str(testIdx), fontsize=16)

    plt.show()


for idx, i in enumerate(files):
    plot_test_pred(test_image=test_image, test_pred=test_pred, testIdx=idx)