# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer
import pandas
from sklearn.preprocessing import LabelEncoder

batch_size = 32

# Input Train data
dataframe = pandas.read_csv('train-articles.csv', header=None, escapechar='\\', na_filter=False)
dataset = dataframe.values
titles = dataset[139:,0]
categories = dataset[139:,0]

# Data representation matrix if word is present 1 or 0 if
tk = Tokenizer(num_words=10000)
tk.fit_on_texts(titles)
x_train = tk.texts_to_matrix(titles, mode='tfidf')

encoder = LabelEncoder()
encoder.fit(categories)
num_categories = encoder.transform(categories)
y_train = keras.utils.to_categorical(num_categories)

# Stack each layer on top of each other
model = Sequential()

# Considered the input layer, dimension of the data
# Generate 512 inputs for next layer
model.add(Dense(512, input_dim=1000))

# Stacking linear operations become linear operation = single layer
# Activation function changes the shape of the learned data
model.add(Activation('relu'))

# Output of the model. Number of categories.
model.add(Dense(7))

# Normalizes the output from the previous layer to be between 0 and 1
model.add(Activation('softmax'))

# categorical_crossentropy is the loss function used when sorting the categories.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

#Train the model
# X_train and Y_Train are the the inputs and outputs.
model.fit(x_train, y_train, batch_size=batch_size, epochs=2, verbose=1, validation_split=0.1)

# Input Train data
dataframe = pandas.read_csv('test-articles.csv', header=None, escapechar='\\', na_filter=false)
dataset = dataframe.values
test_titles = dataset[14:,0]
test_categories = dataset[14:,0]

# Data representation matrix if word is present 1 or 0 if
tk = Tokenizer(num_words=10000)
tk.fit_on_texts(test_titles)
x_test = tk.texts_to_matrix(test_titles, mode='tfidf')

encoder = LabelEncoder()
encoder.fit(test_categories)
num_categories = encoder.transform(test_categories)
y_test = keras.utils.to_categorical(num_categories)

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Score: %2f" % score)
print("Validation Accuracy: %2f" % (acc))
