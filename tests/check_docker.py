
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb

from wandb.keras import WandbCallback
wandb.init(project="my-test-project", entity="victthor")

print("TensorFlow version:", tf.__version__)

# data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10),
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

history = model.fit(
  x_train, 
  y_train,
  batch_size=512,
  epochs=50,
  callbacks=[WandbCallback()],
  )

model.evaluate(x_test,  y_test, verbose=2)

bbb = 1
