from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST datasets for training (60,000 exemplars) and testing (10,000 exemplars)
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_class) = mnist.load_data()
x_train = np.expand_dims(x_train, 3)
x_test = np.expand_dims(x_test, 3)

# Normalize inputs to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = x_train.copy()
y_test = x_test.copy()

# Add noise to test data
noise_level = 0.1
x_test += np.random.normal(0.0, noise_level, size=x_test.shape)

# Model parameters
num_epochs = 5
batch_size = 100
size_input = (28, 28, 1)  # Input shape representing 28x28 pixel intensity values

# Build the autoencoder model
mnist_autoencoder = Sequential([
    keras.Input(shape=size_input, name='input_layer'),
    Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv_encode_1'),
    Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv_encode_2'),
    Conv2D(1, (1, 1), activation='relu', name='conv_encode_3'),
    Conv2DTranspose(16, (3, 3), activation='relu', padding='same', name='conv_decode_1'),
    Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv_decode_2'),
    Conv2DTranspose(1, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv_decode_3')
])

# Print a summary of the model
mnist_autoencoder.summary()

# Compile the model
mnist_autoencoder.compile(
    loss=keras.losses.mse,
    optimizer=keras.optimizers.Adam(),
    metrics=['mse']  # Metrics must be a list
)

# Fit the model
mnist_autoencoder.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(x_test, y_test),
    verbose=2
)

# Dummy call to initialize the model's weights and layers
dummy_data = np.zeros((1, 28, 28, 1))
mnist_autoencoder.predict(dummy_data)

# Create a new model to extract activations from the 'conv_encode_3' layer
layer_name = 'conv_encode_3'

# Create the intermediate model to get activations
intermediate_layer_model = Model(
    inputs=mnist_autoencoder.layers[0].input,  # Access input from the first layer
    outputs=mnist_autoencoder.get_layer(layer_name).output
)

# Get activations for the test data
latent_space_activations = np.squeeze(intermediate_layer_model.predict(x_test))

# Print the shape of the latent space activations
print("Latent space activations shape:", latent_space_activations.shape)
# Visualize the results for 10 randomly selected test samples
random_samples = np.random.randint(0, x_test.shape[0], size=10)

# Generate predictions for original noisy images
y_predict = mnist_autoencoder.predict(x_test[random_samples])

# Apply transformations to the test data
x_test_upside_down = np.flip(x_test[random_samples], axis=1)  # Flip each image vertically (upside down)
x_test_inverted = 1 - x_test[random_samples]  # Invert intensity: dark becomes bright and vice versa

# Get predictions for the transformed images
y_predict_upside_down = mnist_autoencoder.predict(x_test_upside_down)
y_predict_inverted = mnist_autoencoder.predict(x_test_inverted)

# Function to create montages
def create_montage(input_images, ideal_output, predicted_output):
    montage = []
    for i in range(len(input_images)):
        row = np.concatenate(
            (input_images[i].squeeze(), ideal_output[i].squeeze(), predicted_output[i].squeeze()), axis=1
        )
        montage.append(row)
    return np.array(montage)

# Create montages
montage_original = create_montage(x_test[random_samples], y_test[random_samples], y_predict)
montage_upside_down = create_montage(x_test_upside_down, y_test[random_samples], y_predict_upside_down)
montage_inverted = create_montage(x_test_inverted, y_test[random_samples], y_predict_inverted)

# Plot montages
plt.figure(figsize=(15, 5))
plt.title('Original Input (Noisy), Ideal Output, and Reconstructed Output')
plt.imshow(np.vstack(montage_original), cmap='gray')
plt.axis('off')
plt.show()

plt.figure(figsize=(15, 5))
plt.title('Upside Down Input, Ideal Output, and Reconstructed Output')
plt.imshow(np.vstack(montage_upside_down), cmap='gray')
plt.axis('off')
plt.show()

plt.figure(figsize=(15, 5))
plt.title('Inverted Input, Ideal Output, and Reconstructed Output')
plt.imshow(np.vstack(montage_inverted), cmap='gray')
plt.axis('off')
plt.show()

