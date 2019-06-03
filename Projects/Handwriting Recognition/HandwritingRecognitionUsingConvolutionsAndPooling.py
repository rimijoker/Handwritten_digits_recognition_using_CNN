#%%
import tensorflow as tf
print(tf.__version__)
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.999):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()


x_train=x_train.reshape(60000, 28, 28, 1)
x_train=x_train / 255.0
x_test = x_test.reshape(10000, 28, 28, 1)
x_test=x_test/255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=5, callbacks=[callbacks])
model.evaluate(x_test,y_test)

(x_train, y_train),(x_test, y_test) = mnist.load_data()

SingleTestImageIndex = 999
plt.imshow(x_test[SingleTestImageIndex])
plt.show()
print("Testing single image")

x_train=x_train.reshape(60000, 28, 28, 1)
x_train=x_train / 255.0
x_test = x_test.reshape(10000, 28, 28, 1)
x_test=x_test/255.0

classifications = model.predict(x_test)

print(classifications[SingleTestImageIndex])
print(y_test[SingleTestImageIndex])