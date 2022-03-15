from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Dense, Flatten, Dropout

def DNN_model():
    # building model
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(28, activation = 'softmax'))

    # setting optimizer, loss and accuracy matrix
    model.compile(optimizer= 'Adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    return model

def CNN_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(32,32,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    # Add a conv2D layer and MaxPool2D layer
    model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    # Add a conv2D layer and MaxPool2D layer
    model.add(Conv2D(filters=128, kernel_size=7, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))

    # Add a conv2D layer and MaxPool2D layer
    model.add(Conv2D(filters=256, kernel_size=11, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    # Add a conv2D layer and MaxPool2D layer
    model.add(Conv2D(filters=256, kernel_size=11, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.))
    # Add a flatten layer
    model.add(Flatten())
    # Add a Dense layer with 256 unit
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    # Add an output Layer
    model.add(Dense(28,activation='softmax'))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
