import numpy as np
import tensorflow as tf
from tensorflow import keras
import pathlib
import matplotlib.pyplot as plt
import re
import sys
import logging
import os

logging.getLogger("tensorflow").setLevel(logging.DEBUG)

class PostTrainingQuantization:
    def __init__(self, weight_shape=(256, 256)):
        # Initialize the class with random weights of a given shape
        self.weights = np.random.randn(*weight_shape)

    def quantize_and_reconstruct(self, weights):
        """
        Quantizes the given weight matrix from fp32 to int8 and reconstructs it back to fp32.
        """
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        range_val = max_weight - min_weight
        max_int8 = 2 ** 8

        # Compute the scale value
        scale = range_val / max_int8

        # Compute the midpoint
        midpoint = np.mean([max_weight, min_weight])

        # Center and quantize the weights
        centered_weights = weights - midpoint
        quantized_weights = np.rint(centered_weights / scale)

        # Reconstruct the weights back to fp32
        reconstructed_weights = scale * quantized_weights + midpoint
        return reconstructed_weights

    def calculate_max_error(self, original_weights, reconstructed_weights):
        """
        Calculates the maximum error between the original and reconstructed weights.
        """
        errors = reconstructed_weights - original_weights
        max_error = np.max(errors)
        return max_error

    def get_unique_values_count(self, weights):
        """
        Returns the number of unique values in the weight matrix.
        """
        return np.unique(weights).shape[0]

    def compare_weights(self, original_weights, reconstructed_weights):
        """
        Compares the original and reconstructed weights by plotting both along with their errors.
        """
        fig, axs = plt.subplots(3, figsize=(10, 12))

        # Plot original weights
        axs[0].plot(original_weights.flatten(), label="Original Weights", color="blue")
        axs[0].set_title("Original Weights")
        axs[0].legend()

        # Plot reconstructed weights
        axs[1].plot(reconstructed_weights.flatten(), label="Reconstructed Weights", color="green")
        axs[1].set_title("Reconstructed Weights")
        axs[1].legend()

        # Plot error between original and reconstructed weights
        error = original_weights.flatten() - reconstructed_weights.flatten()
        axs[2].plot(error, label="Error (Original - Reconstructed)", color="red")
        axs[2].set_title("Error in Weights")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    def run_quantization(self):
        # Perform quantization and reconstruction
        reconstructed_weights = self.quantize_and_reconstruct(self.weights)

        # Calculate and print the maximum error
        max_error = self.calculate_max_error(self.weights, reconstructed_weights)
        print("Max Error:", max_error)

        # Check the number of unique values in the quantized weights
        unique_values_count = self.get_unique_values_count(reconstructed_weights)
        print("Number of unique values in the quantized matrix:", unique_values_count)

        # Visualize the comparison of original vs reconstructed weights
        self.compare_weights(self.weights, reconstructed_weights)

    def visualize_weight_distribution(self, weights, output_dir='./PostTrainingQuantization_output/'):
        """
        Saves the weight distribution using a histogram in the specified output directory.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.hist(weights.flatten(), bins=100)
        plt.title("Weight Distribution")

        # Define the file path to save the plot
        file_path = os.path.join(output_dir, 'weight_distribution.png')

        # Save the plot instead of showing it
        plt.savefig(file_path)
        plt.close()

        print(f"Weight distribution saved to {file_path}")

    def convert_to_tflite(self, model, output_dir='./PostTrainingQuantization_output/'):
        """
        Converts a Keras model to a TFLite model with and without quantization.
        """
        tflite_models_dir = pathlib.Path(output_dir)
        tflite_models_dir.mkdir(exist_ok=True, parents=True)

        # Convert model to standard TFLite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_model_file = tflite_models_dir / "model.tflite"
        tflite_model_file.write_bytes(tflite_model)

        # Convert model to quantized TFLite format
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()
        tflite_quant_model_file = tflite_models_dir / "model_quant.tflite"
        tflite_quant_model_file.write_bytes(tflite_quant_model)

        return tflite_model_file, tflite_quant_model_file

    def load_mnist_model(self):
        """
        Loads and trains a simple MNIST model using Keras.
        """
        # Load MNIST dataset
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize the input image so that each pixel value is between 0 to 1.
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        # Define the model architecture
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28)),
            keras.layers.Reshape(target_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10)
        ])

        # Train the digit classification model
        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
        return model


# Example of running the class methods
if __name__ == "__main__":
    ptq = PostTrainingQuantization()

    # Visualize the initial weight distribution
    ptq.visualize_weight_distribution(ptq.weights)

    # Run the quantization and reconstruction process, along with comparison plot
    ptq.run_quantization()

    # Load and train the MNIST model
    mnist_model = ptq.load_mnist_model()

    # Convert the model to TFLite
    tflite_model, quant_tflite_model = ptq.convert_to_tflite(mnist_model)

    print(f"Converted models saved at {tflite_model} and {quant_tflite_model}")
