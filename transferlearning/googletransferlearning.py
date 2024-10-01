import numpy as np
import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

class TransferLearningFineTuning:
    def __init__(self):
        self.base_model = None
        self.model = None

    def build_dense_layer(self, units, input_shape=None):
        layer = keras.layers.Dense(units)
        if input_shape:
            layer.build(input_shape)
        return layer

    def build_batchnorm_layer(self, input_shape=None):
        layer = keras.layers.BatchNormalization()
        if input_shape:
            layer.build(input_shape)
        return layer

    def freeze_layer(self, layer):
        layer.trainable = False

    def create_model(self):
        layer1 = keras.layers.Dense(3, activation="relu")
        layer2 = keras.layers.Dense(3, activation="sigmoid")
        model = keras.Sequential([keras.Input(shape=(3,)), layer1, layer2])
        return model, layer1

    def compile_and_train(self, model, layer_to_freeze):
        self.freeze_layer(layer_to_freeze)
        initial_weights = layer_to_freeze.get_weights()

        # Compile and train the model
        model.compile(optimizer="adam", loss="mse")
        model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

        final_weights = layer_to_freeze.get_weights()
        self.compare_weights(initial_weights, final_weights)

    @staticmethod
    def compare_weights(initial_weights, final_weights):
        np.testing.assert_allclose(initial_weights[0], final_weights[0])
        np.testing.assert_allclose(initial_weights[1], final_weights[1])

    def build_transfer_learning_model(self):
        # Load the base model with pre-trained weights
        self.base_model = keras.applications.Xception(
            weights='imagenet',
            input_shape=(150, 150, 3),
            include_top=False
        )
        self.base_model.trainable = False  # Freeze the base model

        # Create a new model on top
        inputs = keras.Input(shape=(150, 150, 3))
        x = self.base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(1)(x)
        self.model = keras.Model(inputs, outputs)

    def compile_transfer_model(self):
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=[keras.metrics.BinaryAccuracy()])

    def train_transfer_model(self, new_dataset, epochs=20, callbacks=None, validation_data=None):
        self.model.fit(new_dataset, epochs=epochs, callbacks=callbacks, validation_data=validation_data)

    def fine_tune_model(self, new_dataset, learning_rate=1e-5, epochs=10, callbacks=None, validation_data=None):
        self.base_model.trainable = True
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                           loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=[keras.metrics.BinaryAccuracy()])
        self.model.fit(new_dataset, epochs=epochs, callbacks=callbacks, validation_data=validation_data)

# Example usage
if __name__ == "__main__":
    # Initialize the transfer learning class
    transfer_learning = TransferLearningFineTuning()

    # Example: Freezing layers and compiling a model
    model, layer1 = transfer_learning.create_model()
    transfer_learning.compile_and_train(model, layer1)

    # Example: Building and training a transfer learning model
    transfer_learning.build_transfer_learning_model()
    transfer_learning.compile_transfer_model()

    # You would need a real dataset here
    # transfer_learning.train_transfer_model(new_dataset)

    # Fine-tune the model if needed
    # transfer_learning.fine_tune_model(new_dataset)
