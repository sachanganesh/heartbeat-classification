from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import regularizers

from sklearn.metrics import roc_auc_score
from hbutils import AUROC_Callback

class HeartbeatClassifier(object):
    def __init__(self, loss, input_shape, num_classes, optimizer=Adam(0.0001), metrics=['accuracy']):
        self.model = Sequential([
            Convolution2D(64, (5, 5), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Convolution2D(32, (5, 5), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(100, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.1)),
            Dropout(0.5),
            BatchNormalization(),
            Dense(50, kernel_initializer='normal'),
            Dropout(0.5),
            BatchNormalization(),
            Dense(8, kernel_initializer='normal'),
            Dropout(0.3),
            BatchNormalization(),
            Dense(num_classes, activation='softmax', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01))
        ])
        
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
    def fit(self, X_train, y_train, validation_data=None, batch_size=4, callbacks=[], epochs=150, verbose=2):
        if verbose > 0:
            auroc = AUROC_Callback((X_train, y_train), validation_data=validation_data)
            callbacks.append(auroc)
        
        history = self.model.fit(X_train, y_train, validation_data=validation_data, batch_size=batch_size, epochs=epochs, shuffle=True, callbacks=callbacks, verbose=verbose)
        
        return history
    
    def evaluate(self, X_test, y_test):
        results_map = {}
        
        self.model.load_weights("best_weights.h5")
        res = self.model.evaluate(X_test, y_test, batch_size=4)
        results_map['loss'] = res[0]
        results_map['acc'] = res[1]
        
        y_prob = self.model.predict_proba(X_test)
        results_map['auroc_score'] = roc_auc_score(y_test, y_prob)

        return results_map