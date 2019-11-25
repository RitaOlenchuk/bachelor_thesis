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


def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def per_pixel_softmax_loss(y_true, y_pred):
    n_classes = 3
    y_true_f = K.reshape(y_true, (-1, n_classes))
    y_pred_f = K.reshape(y_pred, (-1, n_classes))
    return keras.losses.categorical_crossentropy(y_true_f, y_pred_f)

# load json and create model
json_file = open('/usr/local/hdd/rita/DL/model_softmax_hist_640_ZT113_150.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/usr/local/hdd/rita/DL/model_softmax_hist_ZT113_150.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
model.compile(optimizer=Adam(lr=1e-5), loss=per_pixel_softmax_loss, metrics=['accuracy'])
print("After loading model")

#input_shape = (960, 1280, 1)
input_shape = (480, 640, 1)
test_path = "/usr/local/hdd/rita/DL/final_training_set/validate/deletelater/"
n_test_image = 1
endName = "small.tif"


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

test_path_labels = "/usr/local/hdd/rita/DL/final_training_set/validate/deletelater/labels/"
label_shape = (480, 640, 3)
# load test images
test_labels = np.empty(n_test_image * label_shape[0] * label_shape[1] * label_shape[2] )
test_labels = test_labels.reshape((n_test_image, ) + label_shape)

files_lables = sorted([f for f in os.listdir(test_path_labels) if f.endswith(endName)])
files_lables = files_lables[:n_test_image]

patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)

count = 0
for i in files_lables:
    img_path = os.path.join(test_path_labels, str(i))
    img = Image.open(img_path)
    #img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img, dtype=np.float32)
    img = img/np.max(img)

    img [img < 0.2] = 0
    selParts = (img > 0.45) & (img < 0.55)
    img [img >= 0.8] = 1
    img [selParts] = 2
    img [img < 1] = 0

    img = (np.arange(img.max()+1) == img[...,None]).astype(np.float32)
    test_labels[count, :, :, :] = img
    count += 1

test_pred = model.predict(test_image)  

print(test_image.shape, test_labels.shape)
evalu = model.evaluate(x = test_image, y = test_labels, verbose=0)
print(model.metrics_names)
print(evalu)

print(per_pixel_softmax_loss(test_labels[0,:,:,:], test_pred[0,:,:,:]))

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
    #imageio.imwrite('/usr/local/hdd/rita/DL/final_training_set/validate/deletelater/'+str(files[idx])+'_dl.png', result_image[idx, :, :, 0])
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