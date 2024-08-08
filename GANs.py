import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Define the discriminator
def build_discriminator(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 3, activation='relu')(inputs)
    x = Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model

# Define a simpler generator (U-Net)
def build_generator(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D()(conv1)
    
    # Decoder
    up2 = UpSampling2D()(pool1)
    merge2 = concatenate([conv1, up2], axis=3)  # Concatenate along the last axis
    conv2 = Conv2D(32, 3, activation='relu', padding='same')(merge2)
    conv3 = Conv2D(3, 3, activation='relu', padding='same')(conv2)

    model = Model(inputs=inputs, outputs=conv3)
    return model

# Build and compile the discriminator
discriminator = build_discriminator((256, 256, 3))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])

# Build and compile the combined model (generator + discriminator)
generator = build_generator((256, 256, 1))
discriminator.trainable = False
combined = Model(inputs=generator.input, outputs=discriminator(generator.output))
combined.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])

# Load and preprocess data (example: grayscale to color translation)
data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)  # Add validation split if needed
train_data = data_gen.flow_from_directory('D:\Personal\Coding Projects\FYP\lfw-deepfunneled\lfw-deepfunneled', target_size=(256, 256), class_mode=None, batch_size=8, subset='training')

# Training loop
epochs = 10  # Increase the number of epochs
steps_per_epoch = len(train_data)  # Set to the number of batches per epoch

for epoch in range(epochs):
    print(f"Epoch {epoch} starting...")
    
    # Reset the generator to start from the beginning for each epoch
    train_data.reset()

    for i in range(steps_per_epoch):
        print(f"Processing batch {i}...")
        batch = next(train_data)
        real_images = batch
        grayscale_images = tf.image.rgb_to_grayscale(real_images)
        
        # Generate fake color images
        fake_images = generator.predict(grayscale_images)
        
        # Train discriminator
        real_labels = np.ones((batch.shape[0], 1))
        fake_labels = np.zeros((batch.shape[0], 1))
        
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        g_loss = combined.train_on_batch(grayscale_images, real_labels)

        # Display which network is "winning" based on the current losses
        if d_loss[0] < g_loss[0]:
            winner = "Discriminator"
        else:
            winner = "Generator"

        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss[0]}, G Accuracy: {g_loss[1]*100:.2f}%, Winner: {winner}")

    print(f"Epoch {epoch} completed.")

# Save the generator model
generator.save('colorization_generator.h5')
