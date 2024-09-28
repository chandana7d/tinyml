import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pathlib
from tqdm import tqdm


class CatsVsDogsClassifier:
    def __init__(self, batch_size=32, epochs=5):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.train_batches = None
        self.validation_batches = None
        self.test_batches = None
        self.class_names = ['cat', 'dog']
        self.tflite_model_file = None

    def load_data(self):
        setattr(tfds.image_classification.cats_vs_dogs, '_URL',
                "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

        def format_image(image, label):
            image = tf.image.resize(image, (224, 224)) / 255.0
            return image, label

        (raw_train, raw_validation, raw_test), metadata = tfds.load(
            'cats_vs_dogs',
            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True,
            as_supervised=True,
        )

        num_examples = metadata.splits['train'].num_examples
        num_classes = metadata.features['label'].num_classes
        print(f"Number of examples: {num_examples}")
        print(f"Number of classes: {num_classes}")

        self.train_batches = raw_train.shuffle(num_examples // 4).map(format_image).batch(self.batch_size).prefetch(1)
        self.validation_batches = raw_validation.map(format_image).batch(self.batch_size).prefetch(1)
        self.test_batches = raw_test.map(format_image).batch(1)

    def build_model(self):
        module_selection = ("mobilenet_v2", 224, 1280)
        handle_base, pixels, FV_SIZE = module_selection
        MODULE_HANDLE = f"https://tfhub.dev/google/tf2-preview/{handle_base}/feature_vector/4"
        IMAGE_SIZE = (pixels, pixels)

        print(f"Using {MODULE_HANDLE} with input size {IMAGE_SIZE} and output dimension {FV_SIZE}")

        feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                           input_shape=IMAGE_SIZE + (3,),
                                           output_shape=[FV_SIZE],
                                           trainable=False)

        print("Building model with", MODULE_HANDLE)

        self.model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(metadata.features['label'].num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self):
        hist = self.model.fit(self.train_batches, epochs=self.epochs, validation_data=self.validation_batches)

    def save_model(self):
        CATS_VS_DOGS_SAVED_MODEL = "exp_saved_model"
        tf.saved_model.save(self.model, CATS_VS_DOGS_SAVED_MODEL)

        converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)
        tflite_model = converter.convert()
        tflite_models_dir = pathlib.Path("/tmp/")
        self.tflite_model_file = tflite_models_dir / 'model1.tflite'
        self.tflite_model_file.write_bytes(tflite_model)

    def load_tflite_model(self):
        interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model_file))
        interpreter.allocate_tensors()
        return interpreter

    def predict(self, interpreter):
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

        return predictions, test_labels, test_imgs

    def evaluate_predictions(self, predictions, test_labels):
        score = 0
        for item in range(len(predictions)):
            prediction = np.argmax(predictions[item])
            label = test_labels[item]
            if prediction == label:
                score += 1
        print(f"Out of 100 predictions, I got {score} correct")

    def plot_image(self, i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        img = np.squeeze(img)
        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        color = 'green' if predicted_label == true_label else 'red'
        plt.xlabel(
            f"{self.class_names[predicted_label]}: {100 * np.max(predictions_array):.2f}% ({self.class_names[true_label]})",
            color=color)

    def visualize_predictions(self, predictions, test_labels, test_imgs, max_index=73):
        for index in range(max_index):
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            self.plot_image(index, predictions, test_labels, test_imgs)
            plt.show()


if __name__ == "__main__":
    classifier = CatsVsDogsClassifier()
    classifier.load_data()
    classifier.build_model()
    classifier.train_model()
    classifier.save_model()

    interpreter = classifier.load_tflite_model()
    predictions, test_labels, test_imgs = classifier.predict(interpreter)
    classifier.evaluate_predictions(predictions, test_labels)
    classifier.visualize_predictions(predictions, test_labels, test_imgs)
