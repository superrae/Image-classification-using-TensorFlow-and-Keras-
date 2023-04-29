from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np


# loading data and dividing them into two sets, training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# display shapes of all sets
print('shape of x_train: ', x_train.shape)
print('shape of y_train: ', y_train.shape)
print('shape of x_test: ', x_test.shape)
print('shape of y_test: ', y_test.shape)


# display an input pictures
i = 0
plt.imshow(x_train[i], cmap='binary')
plt.show()


# display labels
print(y_train[0])


# encoding labels for both training and test sets
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# display shape of encoded labels
print('shape of encoded y_train: ', y_train_encoded.shape)
print('shape of encoded y_test: ', y_test_encoded.shape)

# display encoded labels
print(y_train_encoded[0])


# reshaping input into 784-dimensional vectors
x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))


# display pixel value
print(set(x_train_reshaped[0]))


# data normalization
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

epsilon = 1e-10

x_train_normalized = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_normalized = (x_test_reshaped - x_mean) / (x_std + epsilon)

# display normalized pixel values
print(set(x_train_normalized[0]))

# creating a model
model = Sequential([
    # two hidden layers, each of which has 128 nodes
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    # output layer with 10 nodes
    Dense(10, activation='softmax')
])

# compiling the model
model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
)
model.summary()

# training the model
model.fit(x_train_normalized, y_train_encoded, epochs=3)

# evaluating the model
loss, accuracy = model.evaluate(x_test_normalized, y_test_encoded)
print('accuracy of model: ', accuracy)


# predictions
predictions = model.predict(x_test_normalized)
print('shape of prediction: ', predictions.shape)

plt.figure(figsize=(12, 12))
start_index = 0

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    prediction = np.argmax(predictions[start_index + i])
    gt = y_test[start_index + i]

    col = 'g'
    if prediction != gt:
        col = 'r'
    plt.xlabel('i={}, prediction={}, gt={}'.format(start_index + i, prediction, gt), color=col)
    plt.imshow(x_test[start_index + i], cmap='binary')
plt.show()

# display probabilities for every class for the wrong prediction at index 8 
plt.plot(predictions[8])
plt.show()
