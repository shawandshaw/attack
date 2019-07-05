from tensorflow import keras
from ConfigAndData import (train_images, train_labels,
                           test_images, test_labels, inputShape)

train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(
    train_images.shape[0], *inputShape)
test_images = test_images.reshape(
    test_images.shape[0], *inputShape)
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)


model = keras.models.load_model('myModel.h5')

batch_size = 128
epochs = 10

model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(test_images, test_labels))

model.save('myModel.h5')
