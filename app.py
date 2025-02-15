from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
import numpy as np
import keras
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

#Loading the dataset
(X_train, Y_train), (X_test, Y_test)  = keras.datasets.reuters.load_data(
    path="reuters.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    test_split=0.2,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
)


#Getting the average length of the sentences for choosing suitable value for max length of the sentences
sent_length = 0
for item in X_train:
    sent_length += 1

sent_length = int(sent_length/len(X_train))
X_train = sequence.pad_sequences(X_train,maxlen=sent_length)
X_test = sequence.pad_sequences(X_test,maxlen=sent_length)

# Convert labels to one-hot encoding
Y_train = to_categorical(Y_train, num_classes=46)
Y_test = to_categorical(Y_test, num_classes=46)


#Vocabulary size
max_features = 100000

# Define the enhanced RNN model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=sent_length))  # Embedding layer

# Adding Bidirectional LSTM layers
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128)))


# Adding dense layers for classification
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(46, activation='softmax'))  # Output layer for multi-class classification


# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train,Y_train,
    epochs=50,batch_size=32,
    validation_split = 0.2,
    verbose = 1
)
