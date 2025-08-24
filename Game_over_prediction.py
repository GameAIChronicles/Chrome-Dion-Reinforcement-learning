import os
import tensorflow as tf
import matplotlib.pyplot as plt

data = tf.keras.utils.image_dataset_from_directory(os.path.join('data', 'img'), image_size=(100, 83), shuffle=True)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
scaled_data = data.map(lambda x, y: (x/225.0, y))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100, 83, 3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(192, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='sigmoid'))

model.compile('adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(scaled_data, epochs=8)


fig = plt.figure()
plt.plot(history.history['accuracy'], color='red', label='Accuracy')
plt.plot(history.history['loss'], color='orange', label='Loss')
plt.legend(loc='upper left')
plt.show()

model.save('done_pre2.h5')