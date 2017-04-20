import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib

matplotlib.use("pdf")
from matplotlib import pyplot as plt

from PIL import Image, ImageChops
import time
import warnings

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import to_categorical

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

def graph_spectrogram(wav_file, save_png=False):
    _, data = get_wav_info(wav_file)
    window = 256
    samp_freq = 256
    plt.specgram(data, window, samp_freq)

    if save_png:
        plt.axis("off")
        plt.specgram(data, window, samp_freq)
        plt.savefig(wav_file.replace("wav", "png"),
                    dpi=100, # Dots per inch
                    frameon=False,
                    aspect="normal",
                    bbox_inches="tight",
                    pad_inches=0) # Spectrogram saved as a .png
        plt.close("all")

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()

    if bbox:
        return im.crop(bbox)

print("==== BEGIN ====")

print("==== PREPROCESS DATA ====")

df = pd.read_csv("./data/set_b.csv")

df = df[pd.notnull(df["label"])]

for i, _ in df.iterrows():
    path = df.ix[i, "fname"]
    path = path.replace("Btraining_", "")

    first_ind = path.find(df.ix[i, "label"])

    path = "./data/wav/" + path[first_ind:]

    if pd.isnull(df.ix[i, "sublabel"]):
        final_ind = path.find("_")

        path = path[:final_ind] + "_" + path[final_ind:]

    df.ix[i, "fname"] = path

print("==== GENERATE SPECTROGRAMS ====")

print("... already generated previously ...")

# global_size = (496, 369)
#
# num_imgs = 0
#
# for i, _ in df.iterrows():
#     path = df.ix[i, "fname"].replace("wav", "png")
#     df.ix[i, "iname"] = path
#     graph_spectrogram(df.ix[i, "fname"], True)
#
#     im = trim(Image.open(path))
#     im.save(path)
#
#     if im.size != global_size:
#         print("Variable Image Size: " + str(i) + ", " + str(im.size) + ", " + str(global_size))
#
#     num_imgs = i
#     time.sleep(0.05)
#
# print("Number of images: ", num_imgs)

print("==== MORE PREPROCESSING ====")

map = {
    "normal": 0,
    "murmur": 1
}

o_df = df

df = pd.DataFrame()
df["image"] = o_df["iname"]
df["label"]  = o_df["label"]

df = df[df.label != "extrastole"]

for i, _ in df.iterrows():
    df.ix[i, "label"] = map[df.ix[i, "label"]]

X = np.array([])
Y = np.array([])

for _, row in df.iterrows():
    img = load_img(row.image)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    if X.size == 0:
        X = x
    else:
        X = np.vstack([X, x])

    y = np.asarray([row.label])
    y.reshape((1,) + y.shape)

    if Y.size == 0:
        Y = y
    else:
        Y = np.vstack([Y, y])

Y = to_categorical(Y)


print("==== TRAINING MODEL ====")

model = Sequential([
    Convolution2D(62, (3, 3), input_shape=(369, 496, 3)),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((3, 3)),
    Convolution2D(31, (3, 3)),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((3, 3)),
    Convolution2D(16, (3, 3)),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((3, 3)),
    Convolution2D(9, (3, 3)),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(800, kernel_initializer="normal"),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.5),
    Dense(200, kernel_initializer="normal"),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.2),
    Dense(100, kernel_initializer="normal"),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.2),
    Dense(2, kernel_initializer="normal"),
    BatchNormalization(),
    Activation("softmax")
])

model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=["accuracy"])

history = model.fit(X, Y, epochs=200, shuffle=True, batch_size=15, validation_split=0.2)

model_path = "./models/model_b/"

model.save(model_path + "model_b.h5")
del model


print("==== BUILDING VISUALS ====")

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_path + "accuracy.png",
            dpi=100, # Dots per inch
            frameon=False,
            aspect="normal",
            bbox_inches="tight",
            pad_inches=0) # Spectrogram saved as a .png
plt.close("all")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_path + "loss.png",
            dpi=100, # Dots per inch
            frameon=False,
            aspect="normal",
            bbox_inches="tight",
            pad_inches=0) # Spectrogram saved as a .png
plt.close("all")

del history

print("==== END ====")
