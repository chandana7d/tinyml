import os
import tempfile
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# Class to handle the TensorFlow model loading, saving, and inference process
class ModelHandler:
    def __init__(self, model_name, image_url, target_size=(224, 224)):
        self.model_name = model_name
        self.target_size = target_size
        self.image_url = image_url
        self.tmpdir = tempfile.mkdtemp()
        self.image_path = None
        self.labels_path = None
        self.pretrained_model = None
        self.loaded_model = None

    def setup_gpu(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def download_image(self):
        # Download the image and convert it for model input
        self.image_path = tf.keras.utils.get_file("image.jpg", self.image_url)
        img = tf.keras.utils.load_img(self.image_path, target_size=self.target_size)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        # Convert image to a numpy array suitable for model input
        x = tf.keras.utils.img_to_array(img)
        x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis, ...])
        return x

    def download_labels(self):
        # Download ImageNet labels
        self.labels_path = tf.keras.utils.get_file(
            'ImageNetLabels.txt',
            'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        imagenet_labels = np.array(open(self.labels_path).read().splitlines())
        return imagenet_labels

    def load_pretrained_model(self):
        # Load the MobileNet pretrained model
        self.pretrained_model = tf.keras.applications.MobileNet()
        return self.pretrained_model

    def infer(self, input_image):
        # Make inference using the pretrained model
        result = self.pretrained_model(input_image)
        return result

    def decode_predictions(self, result, imagenet_labels):
        # Decode the top 5 predictions
        decoded = imagenet_labels[np.argsort(result)[0, ::-1][:5] + 1]
        return decoded

    def save_model(self):
        # Save the model in the SavedModel format
        save_path = os.path.join(self.tmpdir, f"{self.model_name}/1/")
        tf.saved_model.save(self.pretrained_model, save_path)
        print(f"Model saved to {save_path}")
        return save_path

    def load_model(self, model_path):
        # Load a saved model from the SavedModel format
        self.loaded_model = tf.saved_model.load(model_path)
        print(f"Model loaded from {model_path}")
        return self.loaded_model

    def infer_loaded_model(self, input_image):
        # Use the loaded model for inference
        infer_fn = self.loaded_model.signatures["serving_default"]
        result = infer_fn(tf.constant(input_image))[
            self.pretrained_model.output_names[0]
        ]
        return result

# Custom module class for advanced use
class CustomModule(tf.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        self.v = tf.Variable(1.)

    @tf.function
    def __call__(self, x):
        return x * self.v

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def mutate(self, new_v):
        self.v.assign(new_v)

# Main workflow
def main():
    model_handler = ModelHandler(
        model_name="mobilenet",
        image_url="https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
    )

    # Set up GPU for TensorFlow
    model_handler.setup_gpu()

    # Download image and prepare input
    input_image = model_handler.download_image()

    # Download labels
    imagenet_labels = model_handler.download_labels()

    # Load the pretrained MobileNet model
    model_handler.load_pretrained_model()

    # Make inference with the pretrained model
    result_before_save = model_handler.infer(input_image)
    decoded_before = model_handler.decode_predictions(result_before_save, imagenet_labels)
    print("Result before saving:", decoded_before)

    # Save the model
    save_path = model_handler.save_model()

    # Load the saved model
    model_handler.load_model(save_path)

    # Inference with the loaded model
    result_after_load = model_handler.infer_loaded_model(input_image)
    decoded_after = model_handler.decode_predictions(result_after_load, imagenet_labels)
    print("Result after loading:", decoded_after)

if __name__ == "__main__":
    main()
