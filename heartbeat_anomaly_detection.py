import numpy as np
import pandas as pd
from scipy.io import wavfile

from PIL import Image, ImageChops
import time
import warnings

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
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

global_size = (465, 302)

for i, _ in df.iterrows():
    path = df.ix[i, "fname"].replace("wav", "png")
    df.ix[i, "iname"] = path
    graph_spectrogram(df.ix[i, "fname"], True)

    im = trim(Image.open(path))
    im.save(path)

    if im.size != global_size:
        warnings.warn("Variable Image Size: " + str(i) + ", " + str(im.size) + ", " + str(global_size), UserWarning)

    time.sleep(0.05)

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


## AlexNet

# model = Sequential()

# model.add(Convolution2D(64, (3, 3), input_shape=(302, 465, 3)))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(3, 3)))

# model.add(Convolution2D(128, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3, 3)))

# model.add(Convolution2D(192, (6, 6)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3, 3)))

# model.add(Flatten())
# model.add(Dense(4096, kernel_initializer='normal'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(1000, kernel_initializer='normal'))
# model.add(BatchNormalization())
# model.add(Activation('softmax'))

model = Sequential([
    Convolution2D(62, (3, 3), input_shape=(302, 465, 3)),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(2265, kernel_initializer="normal"),
    BatchNormalization(),
    Activation("relu"),
    Dense(2, kernel_initializer="normal"),
    BatchNormalization(),
    Activation("softmax")
])

model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=["accuracy"])

model.output_shape

model.fit(X, Y, epochs=100, shuffle=True, validation_split=0.2)
