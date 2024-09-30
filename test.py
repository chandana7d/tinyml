import tensorflow as tf
import tensorflow_hub as hub

# Define the TensorFlow Hub layer
# Replace 'YOUR_HUB_MODEL_URL' with the actual URL of the TensorFlow Hub model you want to use.
hub_layer = hub.KerasLayer("YOUR_HUB_MODEL_URL", trainable=True)

# Wrap the hub layer in a Lambda layer
hub_layer_wrapper = tf.keras.layers.Lambda(lambda x: hub_layer(x))

# Define the model using a Sequential API
model = tf.keras.Sequential([
    hub_layer_wrapper,  # Use the wrapped layer
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Display the model summary
model.summary()
