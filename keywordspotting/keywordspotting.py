# Import necessary packages
import os
import sys
import numpy as np
import pickle
import tensorflow.compat.v1 as tf
from IPython.display import Audio
import librosa
import scipy.io.wavfile
import ffmpeg

# Clone the TensorFlow Github Repository and set up paths
# !wget https://github.com/tensorflow/tensorflow/archive/v2.14.0.zip
# !unzip v2.14.0.zip &> /dev/null
# !mv tensorflow-2.14.0/ tensorflow/

# Add the speech processing modules path
sys.path.append("/content/tensorflow/tensorflow/examples/speech_commands/")
import input_data
import models

# Configure defaults
WANTED_WORDS = "yes,no"
print("Spotting these words: %s" % WANTED_WORDS)

# Set training constants
number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels + 2  # for 'silence' and 'unknown' labels
equal_percentage_of_training_samples = int(100.0 / number_of_total_labels)
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples

# Constants shared during training and inference
PREPROCESS = 'micro'
WINDOW_STRIDE = 20
MODEL_ARCHITECTURE = 'tiny_conv'

# Directories for data, logs, and models
DATASET_DIR = 'dataset/'
LOGS_DIR = 'logs/'
TRAIN_DIR = 'train/'
MODELS_DIR = 'models'

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

MODEL_TF = os.path.join(MODELS_DIR, 'model.pb')
MODEL_TFLITE = os.path.join(MODELS_DIR, 'model.tflite')
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, 'float_model.tflite')
MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, 'model.cc')
SAVED_MODEL = os.path.join(MODELS_DIR, 'saved_model')

# Quantization constants
QUANT_INPUT_MIN = 0.0
QUANT_INPUT_MAX = 26.0
QUANT_INPUT_RANGE = QUANT_INPUT_MAX - QUANT_INPUT_MIN

# Audio processing constants
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

# URL for the dataset and validation/testing split percentages
DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

# # Loading the pretrained model
# !curl -O "https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_micro_train_2020_05_10.tgz"
# !tar xzf speech_micro_train_2020_05_10.tgz

TOTAL_STEPS = 15000  # Used to identify which checkpoint file

# Generate TensorFlow model for inference
# !rm -rf {SAVED_MODEL}
# !python tensorflow/tensorflow/examples/speech_commands/freeze.py \
# --wanted_words={WANTED_WORDS} \
# --window_stride={WINDOW_STRIDE} \
# --preprocess={PREPROCESS} \
# --model_architecture={MODEL_ARCHITECTURE} \
# --start_checkpoint=MODEL_ARCHITECTURE.ckpt-{TOTAL_STEPS} \
# --save_format=saved_model \
# --output_file={SAVED_MODEL}

# Generate TensorFlow Lite model
model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
    SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
    WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)

audio_processor = input_data.AudioProcessor(
    DATA_URL, DATASET_DIR,
    SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
    WANTED_WORDS.split(','), VALIDATION_PERCENTAGE,
    TESTING_PERCENTAGE, model_settings, LOGS_DIR)

with tf.Session() as sess:
    # Convert to float TFLite model
    float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    float_tflite_model = float_converter.convert()
    float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
    print("Float model is %d bytes" % float_tflite_model_size)

    # Convert to quantized TFLite model
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.lite.constants.INT8
    converter.inference_output_type = tf.lite.constants.INT8

    def representative_dataset_gen():
        for i in range(100):
            data, _ = audio_processor.get_data(1, i * 1, model_settings,
                                                BACKGROUND_FREQUENCY,
                                                BACKGROUND_VOLUME_RANGE,
                                                TIME_SHIFT_MS,
                                                'testing',
                                                sess)
            flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 1960)
            yield [flattened_data]

    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
    print("Quantized model is %d bytes" % tflite_model_size)

# Testing the accuracy after quantization
def run_tflite_inference_testSet(tflite_model_path, model_type="Float"):
    np.random.seed(0)  # Set random seed for reproducible results
    with tf.Session() as sess:
        test_data, test_labels = audio_processor.get_data(
            -1, 0, model_settings, BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
            TIME_SHIFT_MS, 'testing', sess)
        test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

        # Initialize the interpreter
        interpreter = tf.lite.Interpreter(tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        # For quantized models, manually quantize the input data
        if model_type == "Quantized":
            input_scale, input_zero_point = input_details["quantization"]
            test_data = test_data / input_scale + input_zero_point
            test_data = test_data.astype(input_details["dtype"])

        # Evaluate predictions
        correct_predictions = 0
        for i in range(len(test_data)):
            interpreter.set_tensor(input_details["index"], test_data[i])
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])[0]
            top_prediction = output.argmax()
            correct_predictions += (top_prediction == test_labels[i])

        print('%s model accuracy is %f%% (Number of test samples=%d)' % (
            model_type, (correct_predictions * 100) / len(test_data), len(test_data)))

# Compute float and quantized model accuracy
run_tflite_inference_testSet(FLOAT_MODEL_TFLITE)
run_tflite_inference_testSet(MODEL_TFLITE, model_type='Quantized')

# Testing the model on example audio
# !wget --no-check-certificate --content-disposition https://github.com/tinyMLx/colabs/blob/master/yes_no.pkl?raw=true
print("Wait a minute for the file to sync in the Colab and then run the next cell!")

# Load audio files for testing
with open('yes_no.pkl', 'rb') as fid:
    audio_files = pickle.load(fid)
    yes_audio = [audio_files[f'yes{i+1}'] for i in range(4)]
    no_audio = [audio_files[f'no{i+1}'] for i in range(4)]
    sr_yes = [audio_files[f'sr_yes{i+1}'] for i in range(4)]
    sr_no = [audio_files[f'sr_no{i+1}'] for i in range(4)]

# Play the example audio files
for i in range(4):
    print(f"Playing 'yes' audio {i+1}:")
    display(Audio(yes_audio[i], rate=sr_yes[i]))

for i in range(4):
    print(f"Playing 'no' audio {i+1}:")
    display(Audio(no_audio[i], rate=sr_no[i]))

# Test the model on the example files
# !pip install ffmpeg-python &> /dev/null
# !pip install librosa &> /dev/null
# !git clone https://github.com/petewarden/extract_loudest_section.git
# !make -C extract_loudest_section/
# print("Packages Imported, Extract_Loudest_Section Built")

# Helper function to run inference on a single input
TF_SESS = tf.compat.v1.InteractiveSession()

def run_tflite_inference_singleFile(tflite_model_path, custom_audio, sr_custom_audio, model_type="Float"):
    # Preprocess the sample to get the features to pass to the model
    custom_audio_resampled = librosa.resample(custom_audio, sr_custom_audio, SAMPLE_RATE)
    features = audio_processor.extract_features(custom_audio_resampled)
    input_data = features.reshape(1, -1).astype(np.float32)

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # For quantized models, manually quantize the input data
    if model_type == "Quantized":
        input_scale, input_zero_point = input_details["quantization"]
        input_data = input_data / input_scale + input_zero_point
        input_data = input_data.astype(input_details["dtype"])

    # Run inference
    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    top_prediction = output.argmax()

    print('Prediction:', top_prediction)

# Test inference on 'yes' and 'no' audio
for i in range(4):
    print(f"Testing with 'yes' audio {i+1}:")
    run_tflite_inference_singleFile(FLOAT_MODEL_TFLITE, yes_audio[i], sr_yes[i])

for i in range(4):
    print(f"Testing with 'no' audio {i+1}:")
    run_tflite_inference_singleFile(FLOAT_MODEL_TFLITE, no_audio[i], sr_no[i])
