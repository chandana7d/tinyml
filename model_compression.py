# Importing necessary libraries
import tensorflow as tf  # TensorFlow for building and training models
import numpy as np  # NumPy for numerical operations and handling arrays
from tensorflow.keras import Sequential  # Sequential API for creating Keras models
from tensorflow.keras.layers import Dense  # Dense layer for neural networks
import pathlib  # Pathlib for handling file system paths

# Class responsible for creating, training, and managing the TensorFlow model
class ModelTrainer:
    def __init__(self):
        # Initializing variables for the model and the first layer
        self.model = None
        self.layer = None

    def create_model(self):
        # Defining the model architecture with one Dense layer (1 unit, 1 input feature)
        self.layer = Dense(units=1, input_shape=[1])
        # Using Sequential API to create a model with the defined layer
        self.model = Sequential([self.layer])
        # Compiling the model with stochastic gradient descent (sgd) and mean squared error as loss
        self.model.compile(optimizer='sgd', loss='mean_squared_error')

    def train_model(self, xs, ys, epochs=500):
        # Training the model on provided inputs (xs) and outputs (ys) for a set number of epochs
        self.model.fit(xs, ys, epochs=epochs)

    def predict(self, value):
        # Predicting the output for a given input value
        # The input must be reshaped into a 2D NumPy array as expected by Keras
        return self.model.predict(np.array([[value]]))

    def get_weights(self):
        # Returning the trained weights of the Dense layer
        return self.layer.get_weights()

    def save_model(self, export_dir):
        # Saving the trained model in TensorFlow's SavedModel format to the specified directory
        tf.saved_model.save(self.model, export_dir)

    def convert_to_tflite(self, export_dir):
        # Converting the saved TensorFlow model to TensorFlow Lite format (for mobile/embedded devices)
        converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
        return converter.convert()


# Class responsible for handling TensorFlow Lite models, loading, and making predictions
class TFLiteModelHandler:
    def __init__(self, tflite_model):
        # Initializing the TFLite interpreter with the loaded model
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        # Allocating memory for the tensors used by the TFLite interpreter
        self.interpreter.allocate_tensors()
        # Getting details about the model's input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, to_predict):
        # Setting the input tensor with the provided value
        self.interpreter.set_tensor(self.input_details[0]['index'], to_predict)
        # Running the inference on the model
        self.interpreter.invoke()
        # Retrieving the output tensor (the prediction result)
        return self.interpreter.get_tensor(self.output_details[0]['index'])


if __name__ == "__main__":
    # TensorFlow Model Training Workflow
    trainer = ModelTrainer()  # Create an instance of ModelTrainer
    trainer.create_model()  # Define and compile the model

    # Define input (xs) and output (ys) training data
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # Train the model with the given data for 500 epochs
    trainer.train_model(xs, ys)

    # Make a prediction using the trained model
    # The input value is reshaped as required by Keras (2D array)
    prediction = trainer.predict(10.0)
    print(f"Prediction for input 10.0: {prediction}")  # Output the prediction

    # Display the learned weights of the model
    print(f"Model Weights: {trainer.get_weights()}")

    # Save the trained model in the specified directory
    export_dir = 'saved_model/saved_model/run1'
    trainer.save_model(export_dir)

    # Convert the saved model to TensorFlow Lite format
    tflite_model = trainer.convert_to_tflite(export_dir)
    # Save the converted TFLite model as a file
    tflite_model_file = pathlib.Path('model.tflite')
    tflite_model_file.write_bytes(tflite_model)

    # Load and run inference using the TensorFlow Lite model
    handler = TFLiteModelHandler(tflite_model)  # Initialize TFLite model handler

    # Prepare input data for the TFLite model (must be 2D and in float32)
    to_predict = np.array([[10.0]], dtype=np.float32)
    # Make a prediction with the TFLite model
    tflite_result = handler.predict(to_predict)

    # Output the result of the TFLite model prediction
    print(f"TFLite Prediction for input 10.0: {tflite_result}")
