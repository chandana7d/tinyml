import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
import os

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

class MaskDetectionModel:
    def __init__(self):
        self.model = None
        self.data_augmentation = None

    def load_and_preprocess_data(self, dataset_dir):
        """
        Load and preprocess the dataset.
        Arguments:
            dataset_dir: Path to the dataset root folder.
        Returns:
            train_dataset, validation_dataset, test_dataset: Preprocessed dataset objects.
        """
        train_dataset = image_dataset_from_directory(
            os.path.join(dataset_dir, 'train'),
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical',
            shuffle=True
        )

        validation_dataset = image_dataset_from_directory(
            os.path.join(dataset_dir, 'validation'),
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical',
            shuffle=True
        )

        test_dataset = image_dataset_from_directory(
            os.path.join(dataset_dir, 'test'),
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical',
            shuffle=False
        )

        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

        return train_dataset, validation_dataset, test_dataset

    def build_data_augmentation_layer(self):
        """
        Build a data augmentation layer for the model.
        Returns:
            data_augmentation: Data augmentation Sequential model.
        """
        self.data_augmentation = models.Sequential([
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2)
        ])
        return self.data_augmentation

    def build_model(self):
        """
        Build the mask detection model using transfer learning with MobileNetV2.
        """
        # Load MobileNetV2 pre-trained model
        base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
        base_model.trainable = False  # Freeze the base model layers

        # Data augmentation layer
        self.build_data_augmentation_layer()

        # Build the final model
        inputs = layers.Input(shape=IMG_SIZE + (3,))
        x = self.data_augmentation(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)  # Regularization
        outputs = layers.Dense(2, activation='softmax')(x)  # 2 classes: With mask, without mask

        self.model = models.Model(inputs, outputs)

    def compile_model(self):
        """
        Compile the model with appropriate loss function and optimizer.
        """
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self, train_dataset, validation_dataset, epochs=10):
        """
        Train the model with the provided datasets.
        Arguments:
            train_dataset: Preprocessed training dataset.
            validation_dataset: Preprocessed validation dataset.
            epochs: Number of epochs for training.
        """
        self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs
        )

    def evaluate_model(self, validation_dataset):
        """
        Evaluate the model on the validation dataset.
        Arguments:
            validation_dataset: Preprocessed validation dataset.
        """
        loss, accuracy = self.model.evaluate(validation_dataset)
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    def evaluate_on_test(self, test_dataset):
        """
        Evaluate the model on the test dataset.
        Arguments:
            test_dataset: Preprocessed test dataset.
        """
        loss, accuracy = self.model.evaluate(test_dataset)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    def visualize_augmentation(self, dataset, data_augmentation, num_images=5):
        """
        Visualize augmented images.
        Arguments:
            dataset: Dataset to fetch images from.
            data_augmentation: Data augmentation layer.
            num_images: Number of augmented images to visualize.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        for images, _ in dataset.take(1):
            plt.figure(figsize=(10, 10))
            for i in range(num_images):
                augmented_image = data_augmentation(images)[i].numpy().astype("uint8")
                ax = plt.subplot(1, num_images, i + 1)
                plt.imshow(augmented_image)
                plt.axis("off")
            plt.show()


# Main execution
if __name__ == "__main__":
    # Update this to your local dataset directory
    dataset_dir = 'dataset/'  # Update this with your actual dataset path

    # Initialize model object
    mask_model = MaskDetectionModel()

    # Load data
    train_dataset, validation_dataset, test_dataset = mask_model.load_and_preprocess_data(dataset_dir)

    # Build and compile model
    mask_model.build_model()
    mask_model.compile_model()

    # Initial evaluation
    mask_model.evaluate_model(validation_dataset)

    # Train the model
    mask_model.train_model(train_dataset, validation_dataset)

    # Evaluate on test set
    mask_model.evaluate_on_test(test_dataset)

    # Optional: Visualize Data Augmentation
    # mask_model.visualize_augmentation(train_dataset, mask_model.build_data_augmentation_layer())
