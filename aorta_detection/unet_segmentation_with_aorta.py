import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.exposure import equalize_adapthist
import keras
from keras.models import Model
from keras import layers as klayers
from keras.layers import Flatten
from keras.optimizers import Adam
from keras import backend as K
from keras.models import model_from_json


train_path = '/usr/local/hdd/tfunet/aortack/sequences/train'
test_path = '/usr/local/hdd/tfunet/aortack/sequences/test'

n_train_image = 47
n_classes = 3

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# load train images
input_shape = (960, 1280, 1)
#input_shape = (480, 640, 1)

endName = "small.tif"

train_image = np.empty(n_train_image * input_shape[0] * input_shape[1] )
train_image = train_image.reshape((n_train_image, ) + input_shape)
files = sorted([f for f in os.listdir(train_path) if f.endswith(endName)])
count = 0
for i in files:
    img_path = os.path.join(train_path, str(i))
    img = Image.open(img_path)
    img = np.array(img, dtype = np.float32)
    img = rgb2gray(img)
    img /= 255
    img = equalize_adapthist(img)
    train_image[count, :, :, 0] = img[:input_shape[0], :input_shape[1]]
    count += 1
print(len(files))
  
label_shape = (960, 1280, 3)

# load train labels
train_label = np.empty(n_train_image * label_shape[0] * label_shape[1] * label_shape[2] )
train_label = train_label.reshape((n_train_image, ) + label_shape)
files = sorted([f for f in os.listdir(os.path.join(train_path, "labels_plaque")) if f.endswith(endName) or f.endswith("small.tiff")])
print(len(files))

count = 0
for i in files:
    img_path = os.path.join(train_path, "labels_plaque", str(i))
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)
    img = img / np.max(np.max(img))
    #img = equalize_adapthist(img)
    img [img < 0.2] = 0
    selParts = (img > 0.45) & (img < 0.55)
    img [img >= 0.7] = 1
    img [selParts] = 2
    img [img >= 3] = 0
    
    img = (np.arange(img.max()+1) == img[...,None]).astype(np.float32)
    train_label[count, :, :, :] = img
    count += 1

if False:

    for i in range(45, train_image.shape[0]):
        fig = plt.figure(figsize = (12, 12))
        plt.subplot(221)
        plt.imshow(train_image[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.title('Image 0', fontsize=16)
        plt.subplot(222)
        plt.imshow(train_label[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.title('Label 0', fontsize=16)
        plt.show()


def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def per_pixel_softmax_loss(y_true, y_pred):
    y_true_f = K.reshape(y_true, (-1, n_classes))
    y_pred_f = K.reshape(y_pred, (-1, n_classes))
    return keras.losses.categorical_crossentropy(y_true_f, y_pred_f)


def unet(pretrained_weights=None, input_size=input_shape, depth=3, init_filter=8, 
         filter_size=3, padding='same', pool_size=[2, 2], strides=[2, 2]):
    
    inputs = klayers.Input(input_size)
    
    current_layer = inputs
    encoding_layers = []
    
    # Encoder path
    for d in range(depth + 1):
        num_filters = init_filter * 2 ** d
        
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding,
                              kernel_initializer='he_normal')(current_layer)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding,
                              kernel_initializer='he_normal')(conv)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        
        encoding_layers.append(conv)
        
        pool = klayers.MaxPooling2D(pool_size=pool_size)(conv)
        
        if d == depth:
            # Bridge
            current_layer = conv
        else:
            current_layer = pool
    
    # Decoder path
    for d in range(depth, 0, -1):
        num_filters = init_filter * 2 ** d
        up = klayers.Deconvolution2D(num_filters * 2, pool_size,
                                     strides=strides)(current_layer)
        
        crop_layer = encoding_layers[d - 1]
        # Calculate two layers shape
        up_shape = np.array(up._keras_shape[1:-1])
        conv_shape = np.array(crop_layer._keras_shape[1:-1])

        # Calculate crop size of left and right
        crop_left = (conv_shape - up_shape) // 2

        crop_right = (conv_shape - up_shape) // 2 + (conv_shape - up_shape) % 2
        crop_sizes = tuple(zip(crop_left, crop_right))

        crop = klayers.Cropping2D(cropping=crop_sizes)(crop_layer)

        # Concatenate
        up = klayers.Concatenate(axis=-1)([crop, up])
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding,
                              kernel_initializer='he_normal')(up)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding,
                              kernel_initializer='he_normal')(conv)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        
        current_layer = conv
    
    
    outputs = klayers.Conv2D(n_classes, 1, padding=padding,
                             kernel_initializer='he_normal')(current_layer)
    outputs = klayers.Activation('softmax')(outputs)
    model = Model(inputs=inputs, outputs=outputs)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


model = unet(depth=4, filter_size=5)
model.compile(optimizer=Adam(lr=1e-5), loss=per_pixel_softmax_loss) #, metrics=[dice_coefficient, 'accuracy'])
print(model.summary(line_length=135))


history = model.fit(train_image, train_label, batch_size=1, epochs=100, verbose=1, validation_split=0.15)

# serialize model to JSON
model_json = model.to_json()
with open("model_softmax_hist.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_softmax_hist.h5")
print("Saved model to disk")

fig = plt.figure(figsize=(15, 12))

#plt.subplot(221)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

#plt.subplot(222)
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')

plt.show()


