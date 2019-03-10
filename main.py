import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix


def construct_model():
    model = Sequential()
    model.add(Dense(32, input_dim=3))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
             [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 1, 1, 0, 1, 0, 0, 1])

model = construct_model()
optimizer = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.fit(X, y, batch_size=1, epochs=500)

y_ = model.predict(X)
print("Model predictions:\n{}".format(y_))
print("Confusion matrix:\n{}".format(confusion_matrix(y, np.rint(y_))))
