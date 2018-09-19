# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import utils
from keras.preprocessing.text import Tokenizer
import pandas
from sklearn.preprocessing import LabelEncoder

batch_size = 32

# Input Train data
path = '/Users/noemiquezada/Documents/machine-learning/keras-article-classifier/'
dataframe = pandas.read_csv(path + 'train-articles.csv', header=None, escapechar='\\', na_filter=False)
dataset = dataframe.values
titles = dataset[:139,0]
categories = dataset[:139,1]

# Data representation matrix if word is present 1 or 0 if
tk = Tokenizer(num_words=500)
tk.fit_on_texts(titles)
x_train = tk.texts_to_matrix(titles, mode='tfidf')

encoder = LabelEncoder()
encoder.fit(categories)
num_categories = encoder.transform(categories)
y_train = utils.to_categorical(num_categories)
print(y_train)
# Stack each layer on top of each other
model = Sequential()

# Considered the input layer, dimension of the data
# Generate 512 inputs for next layer
model.add(Dense(512, input_dim=500))

# Stacking linear operations become linear operation = single layer
# Activation function changes the shape of the learned data
model.add(Activation('relu'))

# Output of the model. Number of categories.
model.add(Dense(8))

# Normalizes the output from the previous layer to be between 0 and 1
model.add(Activation('softmax'))

# categorical_crossentropy is the loss function used when sorting the categories.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

#Train the model
# X_train and Y_Train are the the inputs and outputs.
model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=1, validation_split=0.1)

# Input Train data
dataframe = pandas.read_csv(path + 'test-articles.csv', header=None, escapechar='\\', na_filter=False)
dataset = dataframe.values
test_titles = dataset[:17,0]
test_categories = dataset[:17,1]

# Data representation matrix if word is present 1 or 0 if
tk_test = Tokenizer(num_words=500)
tk_test.fit_on_texts(test_titles)
x_test = tk_test.texts_to_matrix(test_titles, mode='tfidf')

encoder = LabelEncoder()
encoder.fit(test_categories)
num_categories = encoder.transform(test_categories)
y_test = utils.to_categorical(num_categories)

score, acc = model.evaluate(x_test, y_test, batch_size=32)
print("Score: %2f" % score)
print("Validation Accuracy: %2f" % (acc))
