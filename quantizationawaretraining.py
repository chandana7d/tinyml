import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import tempfile
import os
import pandas as pd  # Import pandas


class QuantizationAwareTrainer:
    def __init__(self):
        self.model = None
        self.q_aware_model = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.results = []  # Store results here

    def load_data(self):
        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
        # Normalize the input images
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def define_model(self):
        # Define the model architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])

    def train_model(self, epochs=1):
        # Compile and train the model
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.fit(self.train_images, self.train_labels, epochs=epochs, validation_split=0.1)

    def apply_quantization(self):
        # Apply quantization aware training
        quantize_model = tfmot.quantization.keras.quantize_model
        self.q_aware_model = quantize_model(self.model)
        self.q_aware_model.compile(optimizer='adam',
                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                   metrics=['accuracy'])

    def fine_tune(self, subset_size=1000, epochs=1):
        # Fine-tune the quantization aware model
        train_images_subset = self.train_images[:subset_size]
        train_labels_subset = self.train_labels[:subset_size]
        self.q_aware_model.fit(train_images_subset, train_labels_subset, batch_size=500, epochs=epochs,
                               validation_split=0.1)

    def evaluate_models(self):
        # Evaluate the baseline and quantized models
        baseline_accuracy = self.model.evaluate(self.test_images, self.test_labels, verbose=0)[1]
        quantized_accuracy = self.q_aware_model.evaluate(self.test_images, self.test_labels, verbose=0)[1]
        return baseline_accuracy, quantized_accuracy

    def create_quantized_model(self):
        # Convert to a quantized TFLite model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite_model = converter.convert()
        return quantized_tflite_model

    def evaluate_tflite_model(self, quantized_tflite_model):
        interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        prediction_digits = []
        for i, test_image in enumerate(self.test_images):
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)
            interpreter.invoke()
            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_digits.append(digit)

        accuracy = (np.array(prediction_digits) == self.test_labels).mean()
        return accuracy

    def model_size_comparison(self, quantized_tflite_model):
        # Create float TFLite model and compare sizes
        float_converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        float_tflite_model = float_converter.convert()

        _, float_file = tempfile.mkstemp('.tflite')
        _, quant_file = tempfile.mkstemp('.tflite')

        with open(quant_file, 'wb') as f:
            f.write(quantized_tflite_model)

        with open(float_file, 'wb') as f:
            f.write(float_tflite_model)

        float_size = os.path.getsize(float_file) / float(2 ** 20)
        quantized_size = os.path.getsize(quant_file) / float(2 ** 20)

        print("Float model size in Mb:", float_size)
        print("Quantized model size in Mb:", quantized_size)

        # Store results in a list
        self.results.append({
            'Model Type': 'Float',
            'Size (MB)': float_size,
            'Accuracy': self.model.evaluate(self.test_images, self.test_labels, verbose=0)[1]
        })

        self.results.append({
            'Model Type': 'Quantized',
            'Size (MB)': quantized_size,
            'Accuracy': self.q_aware_model.evaluate(self.test_images, self.test_labels, verbose=0)[1]
        })

    def save_results_to_csv(self, filename='model_comparison_results.csv'):
        # Save results to a CSV file
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)


# Example of using the class
if __name__ == "__main__":
    trainer = QuantizationAwareTrainer()
    trainer.load_data()
    trainer.define_model()
    trainer.train_model(epochs=1)
    trainer.apply_quantization()
    trainer.fine_tune(subset_size=1000, epochs=1)

    baseline_acc, quant_acc = trainer.evaluate_models()
    print('Baseline test accuracy:', baseline_acc)
    print('Quant test accuracy:', quant_acc)

    quantized_model = trainer.create_quantized_model()
    tflite_accuracy = trainer.evaluate_tflite_model(quantized_model)
    print('TFLite model test accuracy:', tflite_accuracy)

    trainer.model_size_comparison(quantized_model)
    trainer.save_results_to_csv()  # Save the results to a CSV file
