# Deep Stethoscope: CNN Heartbeat Classifier

**[Walkthrough of Data Preprocessing & Model Training](Heartbeat Classification Example.ipynb)**

---

## Objective

As traditional, analog physician tools are reimagined for the digital world, there is an increasing opportunity to bring medical software to care providers and patients. Through these new tools, patient data can now be collected directly and automatically instead of being recorded manually and retroactively. When these modern reinventions are slowly adopted, the platform for medical software grows too.

The stethoscope, for example, is now seeing its electronic counterpart slowly being preferred; these e-stethoscopes provide a number of utilities like ambient noise reduction, audio recording storage, and even remote listening. While these improvements are important, they are only a matter of increasing convenience of use. We can take this technology to the next step by offering insights based on the collected data, at a personal level.

As a proof-of-concept, I've developed a heartbeat audio classifier that is able to differentiate between normal heartbeats and heart murmurs. Though my approach is not perfect, it shows what our modern analytical tools can offer the medical community beyond these incremental technological advances.

## Method

### Audio Preprocessing

The raw data provided is in [WAV](https://en.wikipedia.org/wiki/WAV) format, which is an encoding of an audio. In order for my model to read this information, I convert these heartbeat recordings into an image, a PNG of a 2D [spectrogram](https://en.wikipedia.org/wiki/Spectrogram). Spectrograms are convenient for representing these heartbeat recordings because they capture the intensity of the frequencies throughout a given soundbyte; [recent work](http://deepsound.io/dcgan_spectrograms.html) has shown that one can recreate original audio closely from respective spectrograms, so it can be assumed that spectrograms are effective representations of an audio recording.

These images are then further trimmed to reduce their size, before being fed to the model. In addition, the data is split into stratified training, validation, and testing sets.

### Model Development

The model is a traditional convolutional neural network (CNN) and performs a series of 2D convolutions and max-pooling operations prior to a series of fully connected layers. Due to the class imbalance, Dropout and kernel regularizers are employed selectively to prevent overfitting.

While binary cross-entropy and KL-divergence are effective loss functions for optimizing the accuracy of my model, they fail to properly optimize [AUC-ROC](http://gim.unmc.edu/dxtests/roc3.htm), or the AUROC score. Instead, I employed a loss function that is a differentiable approximation of the AUROC score in order to improve the classifier performance in this regard. While there is a minor decrease in accuracy, there is a significant improvement in the AUROC score of the trained classifier after this change.

### Code

The file [hbutils.py](hbutils.py) contains utility methods for processing the audio and image data. The file [hbmodels.py](hbmodels.py) holds the Keras model architecture. The file [Heartbeat Classification Example.ipynb](Heartbeat Classification Example.ipynb) is a walkthrough of the data preprocessing and model development/training process.

## Results

The CNN achieves an accuracy of approximately 78% on unseen test data, with an approximate AUROC score of 0.77.

## Discussion

While these results are not convincing enough to use the model in a diagnostic or commercial setting, they are still strong and offer a good baseline for future competing methods. The human performance baseline is also unknown, though I suspect the average physician performance is currently better than my model.

There is also the problem of identifying extrasystole heartbeats, of which there is little labeled data. My model only classifies normal heartbeats and murmurs and cannot learn this third classification due to the great class imbalance that occurs.

Although the classifier can definitely be improved, I believe I have established the approach for how to classify heartbeat recordings. Perhaps with better computing resources and more time and data, one can quickly find a better solution with a deeper or more specialized network.
