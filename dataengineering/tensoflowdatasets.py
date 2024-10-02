import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class TFDSLoader:
    """
    A class to handle loading and interacting with TensorFlow Datasets (TFDS).
    It provides functionalities to load datasets, visualize samples, extract dataset metadata,
    and save outputs to an output folder.
    """

    def __init__(self, dataset_name: str, split: str = 'train', as_supervised: bool = False,
                 shuffle_files: bool = True):
        """
        Initialize the TFDSLoader class.

        :param dataset_name: Name of the dataset to load (e.g., 'mnist', 'cifar10').
        :param split: The dataset split to load (default is 'train').
        :param as_supervised: Whether to load the dataset as a tuple (features, label) or a dictionary.
        :param shuffle_files: Whether to shuffle the dataset files during loading.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.as_supervised = as_supervised
        self.shuffle_files = shuffle_files
        self.dataset = None
        self.info = None

    def load_dataset(self, with_info: bool = False):
        """
        Load the dataset using tfds.load.

        :param with_info: Whether to return dataset info along with the dataset.
        """
        self.dataset, self.info = tfds.load(
            self.dataset_name,
            split=self.split,
            as_supervised=self.as_supervised,
            shuffle_files=self.shuffle_files,
            with_info=with_info
        )
        print(f"Loaded dataset: {self.dataset_name}")

    def get_metadata(self):
        """
        Print dataset metadata (e.g., number of classes, image shapes, etc.).
        """
        if self.info is None:
            raise ValueError("Dataset info not loaded. Please set with_info=True when loading the dataset.")
        print("Dataset Metadata:")
        print(self.info)

    def show_examples(self):
        """
        Visualize examples from the dataset using matplotlib.
        """
        if self.dataset is None or self.info is None:
            raise ValueError("Dataset not loaded with info. Please load the dataset with with_info=True.")

        fig = tfds.show_examples(self.dataset, self.info)
        plt.show()

    def iterate_over_dataset(self, num_examples: int = 5):
        """
        Iterate over the dataset and print a few examples.

        :param num_examples: Number of examples to iterate over and display.
        """
        for i, (image, label) in enumerate(self.dataset.take(num_examples)):
            print(f"Example {i + 1}:")
            print(f"Image shape: {image.shape}, Label: {label}")
            plt.imshow(image.numpy().squeeze(), cmap='gray')
            plt.title(f"Label: {label}")
            plt.show()

    def as_numpy(self):
        """
        Convert the dataset to numpy arrays.

        :return: Iterator of numpy arrays.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please load the dataset first.")

        for image, label in tfds.as_numpy(self.dataset):
            print(type(image), type(label), label)

    def save_output(self, num_examples: int = 5, output_folder: str = './data/output'):
        """
        Save a few examples from the dataset as images in the specified output folder.

        :param num_examples: Number of examples to save.
        :param output_folder: The folder where the output will be saved.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please load the dataset first.")

        # Create the output directory if it does not exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Saving output to {output_folder}")

        for i, (image, label) in enumerate(self.dataset.take(num_examples)):
            # Convert the TensorFlow tensor to a numpy array
            image_np = image.numpy()

            # Construct the filename for the image
            file_path = os.path.join(output_folder, f'{self.dataset_name}_example_{i + 1}.png')

            # Save the image
            plt.imsave(file_path, np.squeeze(image_np), cmap='gray')

            print(f"Saved {file_path}")


# Class to Benchmark Datasets
class DatasetBenchmark:
    """
    A class to handle benchmarking TensorFlow Datasets.
    """

    def __init__(self, dataset):
        """
        Initialize the DatasetBenchmark class.

        :param dataset: The dataset to benchmark (tf.data.Dataset).
        """
        self.dataset = dataset

    def benchmark(self, batch_size: int = 32):
        """
        Run the benchmark on the dataset.

        :param batch_size: The batch size for the benchmark.
        """
        ds_batched = self.dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        tfds.benchmark(ds_batched, batch_size=batch_size)


# Main function to use the classes
def main():
    # Initialize the loader for MNIST dataset
    loader = TFDSLoader('mnist', as_supervised=True)

    # Load the dataset with metadata
    loader.load_dataset(with_info=True)

    # Display metadata
    loader.get_metadata()

    # Save some examples to the output folder
    loader.save_output(num_examples=5, output_folder='./data/output')


if __name__ == "__main__":
    main()
