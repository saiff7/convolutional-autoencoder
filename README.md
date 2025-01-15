# convolutional-autoencoder
reconstruct.py:

This code implements and evaluates a convolutional autoencoder using the MNIST dataset. An autoencoder is a neural network designed to encode input data into a lower-dimensional latent space and reconstruct the original data. Here's a breakdown:

1. Data Preparation:
   The MNIST dataset is loaded, which contains handwritten digit images.
   Images are normalized to the range [0, 1].
   Noise is added to the test images to simulate real-world imperfections.

2. Model Architecture:
   A sequential convolutional autoencoder model is built using:
   Encoder: Compresses input images into a latent representation using convolutional layers.
   Decoder: Reconstructs the images from the latent space using transposed convolutional layers.
   The model is summarized, compiled with mean squared error (MSE) loss, and trained using the training data.

3. Intermediate Layer Analysis:
   A separate model is created to extract activations from the encoder's final layer (`conv_encode_3`), representing the compressed latent space.
   Latent space activations for test images are computed and their shape is printed.

4. Image Transformations:
   Test images are manipulated in two ways:
   Flipping vertically (upside-down).
   Inverting pixel intensities.
   The autoencoder generates predictions for both the noisy original and the transformed images.

5. Visualization:
   Montages are created to compare:
   Input images (noisy/transformed).
   Ideal (ground truth) output.
   Reconstructed output from the autoencoder.
   The montages are displayed for visual evaluation of the model's performance.

Purpose:
The goal is to demonstrate the autoencoder's ability to:
Reconstruct noisy images.
Handle transformed images.
Visualize the encoding-decoding process and evaluate the model's robustness.
