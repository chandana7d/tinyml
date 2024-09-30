import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tqdm import tqdm
import pathlib

class CatsVsDogsClassifier:
    def __init__(self, batch_size=32, epochs=5, tflite_model_dir="/tmp/"):
        self.batch_size = batch_size
        self.epochs = epochs
        self.tflite_model_dir = pathlib.Path(tflite_model_dir)
        self.model = None
        self.train_batches = None
        self.validation_batches = None
        self.test_batches = None
        self.num_examples = None
        self.num_classes = None
        self.history = None

    def load_dataset(self):
        def format_image(image, label):
            image = tf.image.resize(image, (224, 224)) / 255.0
            return image, label

        (raw_train, raw_validation, raw_test), metadata = tfds.load(
            'cats_vs_dogs',
            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True,
            as_supervised=True,
        )

        self.num_examples = metadata.splits['train'].num_examples
        self.num_classes = metadata.features['label'].num_classes

        self.train_batches = raw_train.shuffle(self.num_examples // 4).map(format_image).batch(
            self.batch_size).prefetch(1)
        self.validation_batches = raw_validation.map(format_image).batch(self.batch_size).prefetch(1)
        self.test_batches = raw_test.map(format_image).batch(1)

        print(f"Loaded dataset with {self.num_examples} examples and {self.num_classes} classes.")

    def build_model(self):
        # Select the MobileNet V2 model from TensorFlow Hub
        module_handle = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        image_size = (224, 224)  # Input image size for MobileNetV2
        feature_vector_size = 1280  # Output size of the MobileNetV2 feature vector

        print(f"Using {module_handle} with input size {image_size}")

        # Load MobileNetV2 feature vector from TensorFlow Hub
        feature_extractor_layer = hub.KerasLayer(module_handle, input_shape=image_size + (3,), trainable=False)

        # Build the sequential model
        self.model = tf.keras.Sequential([
            feature_extractor_layer,
            tf.keras.layers.Dense(self.num_classes, activation='softmax')  # Classifier layer
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train_model(self):
        if self.model is None or self.train_batches is None:
            print("Model or dataset not initialized.")
            return

        self.history = self.model.fit(self.train_batches, epochs=self.epochs, validation_data=self.validation_batches)
        print("Model training complete.")

    def save_model(self, model_name="exp_saved_model"):
        if self.model is None:
            print("Model not initialized.")
            return

        tf.saved_model.save(self.model, model_name)
        print(f"Model saved as {model_name}.")

    def convert_to_tflite(self, model_version=1, optimize=False, quantize=False):
        model_dir = f"model{model_version}.tflite"
        converter = tf.lite.TFLiteConverter.from_saved_model("exp_saved_model")

        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantize:
            def representative_data_gen():
                for input_value, _ in self.test_batches.take(100):
                    yield [input_value]

            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        tflite_model = converter.convert()
        tflite_model_file = self.tflite_model_dir / model_dir
        tflite_model_file.write_bytes(tflite_model)
        print(f"TFLite model saved as {model_dir}.")

    def test_model_accuracy(self, model_version=1):
        model_dir = f"model{model_version}.tflite"
        interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model_dir / model_dir))
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        predictions = []
        test_labels, test_imgs = [], []

        for img, label in tqdm(self.test_batches.take(100)):
            interpreter.set_tensor(input_index, img)
            interpreter.invoke()
            predictions.append(interpreter.get_tensor(output_index))
            test_labels.append(label.numpy()[0])
            test_imgs.append(img)

        score = sum(np.argmax(pred) == label for pred, label in zip(predictions, test_labels))
        print(f"Out of 100 predictions, {score} were correct.")

        return predictions, test_labels, test_imgs

    def plot_predictions(self, predictions, test_labels, test_imgs, max_index=15):
        class_names = ['cat', 'dog']

        def plot_image(i, predictions_array, true_label, img):
            predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            img = np.squeeze(img)
            plt.imshow(img, cmap=plt.cm.binary)

            predicted_label = np.argmax(predictions_array)
            color = 'green' if predicted_label == true_label else 'red'

            plt.xlabel(
                f"{class_names[predicted_label]} {100 * np.max(predictions_array):2.0f}% ({class_names[true_label]})",
                color=color)

        for index in range(0, max_index):
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plot_image(index, predictions, test_labels, test_imgs)
            plt.show()


# Usage Example
if __name__ == "__main__":
    classifier = CatsVsDogsClassifier()
    classifier.load_dataset()
    classifier.build_model()
    classifier.train_model()
    classifier.save_model()

    classifier.convert_to_tflite(model_version=1, optimize=False)
    classifier.convert_to_tflite(model_version=2, optimize=True)
    classifier.convert_to_tflite(model_version=3, optimize=True, quantize=True)

    predictions, test_labels, test_imgs = classifier.test_model_accuracy(model_version=1)
    classifier.plot_predictions(predictions, test_labels, test_imgs)
