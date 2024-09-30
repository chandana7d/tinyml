import os
import shutil
import tensorflow as tf
import datetime


class MNISTModel:
    def __init__(self):
        # Initialize the model and data attributes
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

        # Initialize metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.EPOCHS = 5

        # Load and prepare data
        self.load_data()

        # Create the model
        self.create_model()

    def load_data(self):
        """Load and normalize the MNIST dataset."""
        print("Loading and normalizing MNIST dataset...")
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train, self.x_test = x_train / 255.0, x_test / 255.0
        self.y_train, self.y_test = y_train, y_test

        print(f"Number of x_train samples: {len(x_train)}, Number of y_train samples: {len(y_train)}")
        print(f"Number of x_test samples: {len(x_test)}, Number of y_test samples: {len(y_test)}")
        # Create TensorFlow datasets for training and testing
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(60000).batch(64)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(64)

    def create_model(self):
        """Create and compile the Keras model."""
        print("Creating the model...")
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28), name='layers_flatten'),
            tf.keras.layers.Dense(512, activation='relu', name='layers_dense'),
            tf.keras.layers.Dropout(0.2, name='layers_dropout'),
            tf.keras.layers.Dense(10, activation='softmax', name='layers_dense_2')
        ])

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_object,
                           metrics=['accuracy'])

    def train_step(self, x_batch, y_batch):
        """Perform a single training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            loss = self.loss_object(y_batch, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update metrics
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(y_batch, predictions)

    def test_step(self, x_batch, y_batch):
        """Perform a single testing step."""
        predictions = self.model(x_batch)
        loss = self.loss_object(y_batch, predictions)

        # Update metrics
        self.test_loss.update_state(loss)
        self.test_accuracy.update_state(y_batch, predictions)

    def train(self):
        """Train the model and log the results."""
        print("Starting training...")
        # Set up TensorBoard logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for epoch in range(self.EPOCHS):
            # Training loop
            for (x_train_batch, y_train_batch) in self.train_dataset:
                self.train_step(x_train_batch, y_train_batch)

            # Log training metrics
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)

            # Testing loop
            for (x_test_batch, y_test_batch) in self.test_dataset:
                self.test_step(x_test_batch, y_test_batch)

            # Log testing metrics
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)

            # Print epoch results
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result() * 100,
                                  self.test_loss.result(),
                                  self.test_accuracy.result() * 100))


# Example usage
if __name__ == "__main__":
    # Clear any logs from previous runs
    log_dir = './logs/'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # Remove the logs directory

    # Instantiate and train the model
    mnist_model = MNISTModel()
    mnist_model.train()

    # Uncomment this line to run TensorBoard in Jupyter Notebook
    # %tensorboard --logdir logs/gradient_tape
