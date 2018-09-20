# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import utils
from keras.preprocessing.text import Tokenizer
import pandas
from sklearn.preprocessing import LabelBinarizer

batch_size = 32
vocab_size = 1000
num_labels = 8

# Input Train data
path = '/Users/noemiquezada/Documents/machine-learning/keras-article-classifier/'
dataframe = pandas.read_csv(path + 'train-articles.csv', header=None, escapechar='\\', na_filter=False)
dataset = dataframe.values
titles = dataset[:139,0]
categories = dataset[:139,1]

# Data representation matrix if word is present 1 or 0 if
tk = Tokenizer(num_words=vocab_size)
tk.fit_on_texts(titles)
x_train = tk.texts_to_matrix(titles, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(categories)
y_train = encoder.transform(categories)

# Stack each layer on top of each other
model = Sequential()

# Considered the input layer, dimension of the data
# Generate 512 inputs for next layer
model.add(Dense(512, input_shape=(vocab_size,)))

# Stacking linear operations become linear operation = single layer
# Activation function changes the shape of the learned data
model.add(Activation('relu'))

# Output of the model. Number of categories.
model.add(Dense(num_labels))

# Normalizes the output from the previous layer to be between 0 and 1
model.add(Activation('softmax'))

print(model.summary())

# categorical_crossentropy is the loss function used when sorting the categories.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model
# X_train and Y_Train are the the inputs and outputs.
model.fit(x_train, y_train, batch_size=32, epochs=30, verbose=1, validation_split=0.1)

# Input Train data
dataframe = pandas.read_csv(path + 'test-articles.csv', header=None, escapechar='\\', na_filter=False)
dataset = dataframe.values
test_titles = dataset[:14,0]
test_categories = dataset[:14,1]

# Data representation matrix if word is present 1 or 0 if
tk_test = Tokenizer(num_words=vocab_size)
tk_test.fit_on_texts(test_titles)
x_test = tk_test.texts_to_matrix(test_titles, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(test_categories)
y_test = encoder.transform(test_categories)

score, acc = model.evaluate(x_test, y_test, batch_size=32)
print("Score: %2f" % score)
print("Validation Accuracy: %2f" % (acc))

category_labels = encoder.classes_

for i in range(len(test_titles)):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = category_labels[np.argmax(prediction[0])]
    print(dataframe.iloc[i,0])
    print('Actual label:' + dataframe.iloc[i,1])
    print("Predicted label: " + predicted_label)
