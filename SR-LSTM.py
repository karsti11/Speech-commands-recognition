import os
import json
import numpy as np 
import random
import librosa
import pandas as pd
from scipy.io import wavfile
import scipy.signal
import matplotlib.pyplot as plt
from scipy import signal
from random import shuffle
from math import floor
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm
import itertools

#from tensorflow.keras.utils import to_categorical

path = 'D:/ML Python/Programi/SpeechCommands/'

dirs = os.listdir(path)
train_df = pd.read_csv('D:/ML Python/Programi/SR/train_labels.csv')
val_df = pd.read_csv('D:/ML Python/Programi/SR/val_labels.csv')
train_df = train_df[:5489]
val_df = val_df[:970]

train_df = train_df.sample(frac=1)
val_df = val_df.sample(frac=1)
# train_labels = to_categorical(train_labels, num_classes=3)
# validation_labels = to_categorical(validation_labels, num_classes=3)		
train_list = train_df['Filename'].tolist()
#train_list = [x for x in train_list if "nohash" in x]
validation_list = val_df['Filename'].tolist()
#validation_list = [x for x in validation_list if "nohash" in x]
#samples,sample_rate = librosa.load(path + 'yes/0c5027de_nohash_1.wav')
#S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

train_labels = train_df['Label'].tolist()
validation_labels = val_df['Label'].tolist()

print(len(train_list))
print(len(validation_list))
print(len(train_labels))
print(len(validation_labels))

sample_rate = 16000

def load_and_process(audio_path):
	
	sr, audio_data = wavfile.read(audio_path)
	audio_signal = signal.resample(audio_data, sample_rate)
	audio_signal = audio_signal/np.amax(audio_signal)
	zcr_signal = librosa.feature.zero_crossing_rate(audio_signal, frame_length=512, hop_length=410) #(1,32)
	mfcc_signal = librosa.feature.mfcc(audio_signal,sample_rate, n_fft=1024, hop_length=410) # (20,32)
	centr_signal = librosa.feature.spectral_centroid(audio_signal, sample_rate, n_fft=1024, hop_length=410)# (1,32)
	rolloff_signal = librosa.feature.spectral_rolloff(audio_signal,n_fft=1024, hop_length=410) # (1,32)
	
	features_mat = np.concatenate((zcr_signal, mfcc_signal, centr_signal, rolloff_signal), axis=0)
	return features_mat


train_data = np.zeros((len(train_list), 23, 40))
validation_data = np.zeros((len(validation_list), 23, 40))

for i in tqdm(range(0,len(train_list))):
	train_data[i,:,:] = load_and_process(path + train_list[i])

for i in tqdm(range(0,len(validation_list))):
	validation_data[i,:,:] = load_and_process(path + validation_list[i])


def build_model():
	model = Sequential()
	model.add(layers.LSTM(64,input_shape=(train_data.shape[1:]), return_sequences=True))
	model.add(layers.Dropout(0.4))
	model.add(layers.LSTM(128))
	model.add(layers.Dropout(0.4))
	model.add(layers.Dense(32, activation='relu'))
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(3, activation='softmax'))
	#model.add(layers.Dropout(0.2))

	model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
	return model

model = build_model()
model.summary()

checkpoint_path = "D:/ML Python/Programi/SR/weights-best.hdf5"
cp_callback = ModelCheckpoint(checkpoint_path,
				monitor ='val_loss',
				verbose = 1,
                                save_best_only=True,                               
                                #error
                                mode = 'min')

callbacks_list = [cp_callback]
history = model.fit(train_data, train_labels,
					validation_data=(validation_data, validation_labels),
 					epochs=70,					
 					callbacks = callbacks_list,
 					)

model.load_weights('path/weights-best.hdf5')

print(train_data.shape)
print(validation_data.shape)

#Plot training & validation accuracy
history_dict = history.history
print(history_dict.keys())
plt.figure(1)
plt.plot(history_dict['accuracy'])
plt.plot(history_dict['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train data', 'Test data'], loc='upper left')
plt.grid(True)
#plt.show()

# Plot training & validation loss values
plt.figure(2)
plt.plot(history_dict['loss'])
plt.plot(history_dict['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train data', 'Test data'], loc='upper left')
plt.grid(True)
plt.show()

val_predict = model.predict(validation_data)

print(type(val_predict))
print(type(validation_labels))
validation_labels = np.asarray(validation_labels)
val_predict = val_predict.argmax(axis=1)
print(val_predict)
print(confusion_matrix(validation_labels,val_predict, labels = [0,1,2]))
target_names = ['marvin','yes','no']
print(classification_report(validation_labels,val_predict, target_names=target_names))

# For more examples of confusion matrix plot, you can check here: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/48018785
def plot_confusion_matrix(y_true, y_pred, 
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['marvin','no','yes']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(validation_labels, val_predict, normalize=True,
                      title='Normalized confusion matrix')
#classes=np.asarray([0,1,2])
plt.show()
target_names = ['marvin','yes','no']
classif_report = classification_report(validation_labels,val_predict, target_names=target_names)
print(classif_report)
