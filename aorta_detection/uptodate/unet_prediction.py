import os
import numpy as np
from PIL import Image
import imageio
from matplotlib import pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_nl_means, estimate_sigma
import cv2
import keras
from keras.models import Model
from keras import layers as klayers
from keras.optimizers import Adam
from keras import backend as K
from keras.models import model_from_json


def per_pixel_softmax_loss(y_true, y_pred):
    n_classes = 3
    y_true_f = K.reshape(y_true, (-1, n_classes))
    y_pred_f = K.reshape(y_pred, (-1, n_classes))
    return keras.losses.categorical_crossentropy(y_true_f, y_pred_f)

# load json and create model
json_file = open('/usr/local/hdd/rita/DL/model_softmax_hist_640_ZT113.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/usr/local/hdd/rita/DL/model_softmax_hist_ZT113.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
model.compile(optimizer=Adam(lr=1e-5), loss=per_pixel_softmax_loss, metrics=['accuracy'])
print("After loading model")

#input_shape = (960, 1280, 1)
input_shape = (480, 640, 1)
test_path = "/usr/local/hdd/rita/DL/sequence/"
n_test_image = 8
endName = "small.tif"

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# load test images
test_image = np.empty(n_test_image * input_shape[0] * input_shape[1] )
test_image = test_image.reshape((n_test_image, ) + input_shape)

files = sorted([f for f in os.listdir(test_path) if f.endswith(endName) and f.startswith("ZT1")])
files = files[:n_test_image]

patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)

count = 0
for i in files:
    img_path = os.path.join(test_path, str(i))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img, dtype=np.float32)
    img = img/np.max(img)
    test_image[count, :, :, 0] = img[:input_shape[0], :input_shape[1]]
    count += 1


test_pred = model.predict(test_image)

result_image = np.empty(n_test_image * input_shape[0] * input_shape[1])
result_image = result_image.reshape((n_test_image, ) + input_shape)
for i in range(n_test_image):
    predicted = test_pred[i,:,:,:]
    tmp = np.zeros((predicted.shape[0], predicted.shape[1]), dtype=int)
    for r in range(tmp.shape[0]):
        for c in range(tmp.shape[1]):
            tmp[r,c] = np.argmax(predicted[r,c])
    result_image[i,:,:] = tmp.reshape(input_shape)

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

size = len(files)
for idx, i in enumerate(files):
    imageio.imwrite('/usr/local/hdd/rita/DL/model_ZT113_100/'+str(files[idx])+'_dl.png', result_image[idx, :, :, 0])
    #plot_test_pred(test_image=test_image, test_pred=result_image, testIdx=idx)
    plt.imshow(test_pred[idx,:,:,:])
    plt.show()
  


fig = plt.figure()
grid = plt.GridSpec(size, 2, wspace=0.1, hspace=0.35)

for idx, i in enumerate(files):
    ax = fig.add_subplot(grid[idx, 0])
    ax.imshow(test_image[idx, :, :, 0], cmap='gray')
    ax.set_title('Test ' + str(idx), fontsize=16)

    ax = fig.add_subplot(grid[idx, 1])
    ax.imshow(result_image[idx, :, :, 0], cmap='gray')
    ax.set_title('Predict ' + str(idx), fontsize=16)
plt.show()