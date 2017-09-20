from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from keras import backend as K
from keras.utils import plot_model
import matplotlib.pyplot as plt


class MNISTCNN(object):
    def __init__(self, batch_size, num_classes, epochs):
        super(MNISTCNN, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs

        # input image dimensions
        self.img_rows, self.img_cols = 28, 28

        self.load_data()
        self.load_model()

    def load_data(self):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1,self. img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

        return (self.x_train, self.y_train, self.x_test, self.y_test)

    def load_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self.model = model
        return self.model

    def fit_model(self):
        train_history = self.model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  validation_data=(self.x_test, self.y_test))

        return train_history

    def evaluate_model(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def run(self):
        train_history = self.fit_model()
        self.evaluate_model()

        return train_history

class MNISTCNNDSC(MNISTCNN):
    def load_model(self):
        model = Sequential()
        model.add(SeparableConv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(SeparableConv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self.model = model
        return self.model

def plot_history(history_cnn, history_cnn_dsc):
    cnn_loss = history_cnn.history['loss']
    cnn_val_loss = history_cnn.history['val_loss']

    dsc_cnn_loss = history_cnn_dsc.history['loss']
    dsc_cnn_val_loss = history_cnn_dsc.history['val_loss']

    plt.plot(cnn_loss)
    plt.plot(cnn_val_loss)

    plt.plot(dsc_cnn_loss)
    plt.plot(dsc_cnn_val_loss)

    plt.legend(['cnn_loss', 'cnn_val_loss', 'dsc_cnn_loss', 'dsc_cnn_val_loss'])

    plt.savefig('mnist_cnn_dsc_loss.png')
    plt.show()


def main():
    mnist_cnn_dsc = MNISTCNNDSC(128, 10, 12)
    mnist_cnn_dsc.model.summary()
    plot_model(mnist_cnn_dsc.model, to_file='mnist_cnn_dsc.png', show_shapes=True)

    mnist_cnn = MNISTCNN(128, 10, 12)
    mnist_cnn.model.summary()
    plot_model(mnist_cnn.model, to_file='mnist_cnn.png', show_shapes=True)

    train_history_cnn = mnist_cnn.run()
    train_history_cnn_dsc = mnist_cnn_dsc.run()





if __name__ == '__main__':
    main()
