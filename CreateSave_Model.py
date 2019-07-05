from tensorflow import keras
from tensorflow.python.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                            MaxPooling2D)
from ConfigAndData import (inputShape, classLabelNames)


# 新的模型
model = keras.Sequential()
model.add(Conv2D(32,
                 activation='relu',
                 input_shape=inputShape,
                 kernel_size=(3, 3),
                 padding='same'))
model.add(Conv2D(64, activation='relu',
                 kernel_size=(3, 3),
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))
# model.add(Flatten(input_shape=inputShape))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classLabelNames), activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
print('Model has been created and saved. \n')
model.save('myModel.h5')
