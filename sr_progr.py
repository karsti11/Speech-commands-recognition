import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import librosa
import scipy.io as sio

train_data_load = sio.loadmat('train_data.mat')
train_data = train_data_load['train_data']

train_labels_load = sio.loadmat('train_labels.mat')
train_labels = train_labels_load['train_labels']

train_labels = np.where(train_labels == 1, 0, train_labels)
train_labels = np.where(train_labels == 2, 1, train_labels)
train_labels = np.where(train_labels == 3, 2, train_labels)


test_data_load = sio.loadmat('test_data.mat')
test_data = test_data_load['test_data']

test_labels_load = sio.loadmat('test_labels.mat')
test_labels = test_labels_load['test_labels']

test_labels = np.where(test_labels == 1, 0, test_labels)
test_labels = np.where(test_labels == 2, 1, test_labels)
test_labels = np.where(test_labels == 3, 2, test_labels)


zcr_train_matrix = np.zeros((5198,1,32))
mfcc_train_matrix = np.zeros((5198,20,32))
chroma_train_matrix = np.zeros((5198,12,32))
cent_train_matrix = np.zeros((5198,1,32))
rolloff_train_matrix = np.zeros((5198,1,32))


print("Train dataset features extracting...")

for i in range(0,5198):

	zcr_example_train = librosa.feature.zero_crossing_rate(train_data[i,:])
	zcr_train_matrix[i,:,:] = zcr_example_train

	mfcc_example_train = librosa.feature.mfcc(train_data[i,:],16000)
	mfcc_train_matrix[i,:,:] = mfcc_example_train/np.amax(mfcc_example_train)

	chroma_example_train = librosa.feature.chroma_stft(train_data[i,:],16000)
	chroma_train_matrix[i,:,:] = chroma_example_train

	cent_example_train = librosa.feature.spectral_centroid(train_data[i,:], 16000)
	cent_train_matrix[i,:,:] = cent_example_train/np.amax(cent_example_train)

	rolloff_example_train = librosa.feature.spectral_rolloff(train_data[i,:],16000)
	rolloff_train_matrix[i,:,:] = rolloff_example_train/np.amax(rolloff_example_train)


features_train_matrix = np.concatenate((zcr_train_matrix, mfcc_train_matrix, chroma_train_matrix, cent_train_matrix,
 rolloff_train_matrix), axis=1)

print(features_train_matrix.shape)


zcr_test_matrix = np.zeros((1300,1,32))
chroma_test_matrix = np.zeros((1300,12,32))
cent_test_matrix = np.zeros((1300,1,32))
rolloff_test_matrix = np.zeros((1300,1,32))
mfcc_test_matrix = np.zeros((1300,20,32))

print("Test dataset features extracting...")

for i in range(0,1300):

	mfcc_example_test = librosa.feature.mfcc(test_data[i,:])
	mfcc_test_matrix[i,:,:] = mfcc_example_test/np.amax(mfcc_example_test)

	zcr_example_test = librosa.feature.zero_crossing_rate(test_data[i,:])
	zcr_test_matrix[i,:,:] = zcr_example_test

	chroma_example_test = librosa.feature.chroma_stft(test_data[i,:],16000)
	chroma_test_matrix[i,:,:] = chroma_example_test

	cent_example_test = librosa.feature.spectral_centroid(test_data[i,:], 16000)
	cent_test_matrix[i,:,:] = cent_example_test/np.amax(cent_example_test)

	rolloff_example_test = librosa.feature.spectral_rolloff(test_data[i,:],16000)
	rolloff_test_matrix[i,:,:] = rolloff_example_test/np.amax(rolloff_example_test)


features_test_matrix = np.concatenate((zcr_test_matrix,mfcc_test_matrix, chroma_test_matrix, cent_test_matrix,
 rolloff_test_matrix), axis=1)


print(features_test_matrix.shape)
print("Features extraction done.")



model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(256,input_shape=(features_train_matrix.shape[1:]), return_sequences=True),
  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.LSTM(128,activation ='relu'),
  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(3, activation='softmax'),
  tf.keras.layers.Dropout(0.2),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Model fitting...")

model.fit(features_train_matrix, train_labels, epochs=30, validation_data=(features_test_matrix, test_labels))
