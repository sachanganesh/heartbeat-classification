import pathlib
import os
import glob
import time

import zipfile

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from PIL import Image, ImageChops

import librosa
import librosa.display

import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def unzip_data(zip_path, target_path):
    target_path = os.path.normpath(target_path)
    pathlib.Path(target_path).mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)
        
def load_wavfile(wavfile_path, sr=1000):
    return librosa.core.load(wavfile_path, sr=1000, mono=True)
    
def get_spectrogram_from_wav(wavfile_path, window_size_sec=0.96, window_shift_sec=0.01, sr=1000, hop_length=64):
    data, _ = load_wavfile(wavfile_path, sr=sr)
    
    win_length = int(sr * window_size_sec)
    # hop_length = int(sr * window_shift_sec)
    
    nfft = win_length
    
    spec = librosa.core.stft(data, n_fft=nfft, hop_length=hop_length)
    spec = librosa.feature.melspectrogram(S=spec)
    
    return spec

def plot_spectrogram(spectrogram, out_path=None):
    ax = librosa.display.specshow(spectrogram)
    
    if out_path is None:
        ax.plot()
    else:
        plt.axis('off')
        plt.savefig(out_path, dpi=100, frameon=False, aspect='normal', bbox_inches='tight', pad_inches=0)
        plt.close('all')
        
def draw_spectrogram_png(spectrogram_path):
    img = mpimg.imread(spectrogram_path)
    plt.imshow(img)
    
def trim_png(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()

    if bbox:
        return image.crop(bbox)

def convert_wavfiles_to_spectrograms(wavfiles_path, out_path, exists_ok=True, trim=0, verbose=False):
    out_path = os.path.normpath(out_path)
    pathlib.Path(out_path).mkdir(exist_ok=True)
    
    wavfiles_path = os.path.normpath(wavfiles_path)
    for wavfile in glob.glob(wavfiles_path + '/*.wav'):
        specfile = out_path + '/' + os.path.basename(wavfile).replace('wav', 'png')
        
        if not exists_ok or not os.path.isfile(specfile):
            spec = get_spectrogram_from_wav(wavfile)
            plot_spectrogram(spec, out_path=specfile)
            
            while trim > 0:
                im = Image.open(specfile)
                im = trim_png(im)
                im.save(specfile)
                trim -= 1
            
            if verbose:
                print(specfile)
            
            time.sleep(0.05)
            
def spectrogram_generator(wavfiles_list):
    for wavfile in wavfiles_list:
        yield get_spectrogram_from_wav(wavfile)
        
def load_df(csv_path, wavfiles_path, specfiles_path):
    wavfiles_path = os.path.normpath(wavfiles_path)
    specfiles_path = os.path.normpath(specfiles_path)
    
    df = pd.read_csv(csv_path)
    df = df[pd.notnull(df['label'])]
    
    for i, _ in df.iterrows():
        path = df.loc[i, 'fname']
        path = path.replace('Btraining_', '')

        first_ind = path.find(df.loc[i, 'label'])

        path = path[first_ind:]

        if pd.isnull(df.loc[i, 'sublabel']):
            final_ind = path.find('_')

            path = path[:final_ind] + '_' + path[final_ind:]

        df.loc[i, 'fname'] = wavfiles_path + '/' + path
        df.loc[i, 'iname'] = specfiles_path + '/' + os.path.basename(path).replace('wav', 'png')
        
    return df

def load_spec_df(df, label_map, exclude_labels=[]):
    spec_df = pd.DataFrame()
    spec_df['spectrogram'] = df['iname']
    spec_df['label'] = df['label']
    
    for i, _ in spec_df.iterrows():
        spec_df.loc[i, "label"] = label_map[spec_df.loc[i, "label"]]
    
    spec_df = spec_df[~spec_df.label.isin(exclude_labels)]
    return spec_df

def get_train_test_validation_split(df, test_size=0.2, val_size=0.2, random_state=7):
    r_df = df.sample(frac=1).reset_index(drop=True)

    X = np.array([])
    Y = np.array([])

    for _, row in r_df.iterrows():
        img = load_img(row.spectrogram)
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

    X_t, X_val, y_t, y_val = train_test_split(X, Y, test_size=val_size, stratify=Y, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=test_size, stratify=y_t, random_state=random_state)
    return X_train, X_test, X_val, y_train, y_test, y_val

def pair_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]
    y_pos = tf.expand_dims(y_pos, 0)
    y_neg = tf.expand_dims(y_neg, -1)
    out = tf.sigmoid(y_neg - y_pos)
    return tf.reduce_mean(out)

class AUROC_Callback(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data=None):
        self.x = training_data[0]
        self.y = training_data[1]
        
        if validation_data is not None:
            self.x_val = validation_data[0]
            self.y_val = validation_data[1]
        else:
            self.x_val = None
            self.y_val = None
        
    
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):        
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        print('\rroc-auc: {}'.format(str(round(roc,4)), end=100 * ' ' + '\n'))
        
        if self.x_val is not None:
            y_pred_val = self.model.predict(self.x_val)
            roc_val = roc_auc_score(self.y_val, y_pred_val) 
            print('\rroc-auc_val: {}'.format(str(round(roc_val,4)), end=100 * ' ' + '\n'))
        
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return