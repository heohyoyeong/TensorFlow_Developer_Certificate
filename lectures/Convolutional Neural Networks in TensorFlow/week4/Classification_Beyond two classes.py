# grader-required-cell

import csv
import string
import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

# sign_mnist_train.csv
TRAINING_FILE = './sign_mnist_train.csv'
VALIDATION_FILE = './sign_mnist_test.csv'

# grader-required-cell

with open(TRAINING_FILE) as training_file:
  line = training_file.readline()
  print(f"First line (header) looks like this:\n{line}")
  line = training_file.readline()
  print(f"Each subsequent line (data points) look like this:\n{line}")


# grader-required-cell

# GRADED FUNCTION: parse_data_from_input
def parse_data_from_input(filename):
    """
    Parses the images and labels from a CSV file

    Args:
      filename (string): path to the CSV file
      d
    Returns:
      images, labels: tuple of numpy arrays containing the images and labels
    """
    with open(filename) as file:
        ### START CODE HERE

        # Use csv.reader, passing in the appropriate delimiter
        # Remember that csv.reader can be iterated and returns one line in each iteration
        csv_reader = csv.reader(file, delimiter=",")
        # np.array
        i=0
        labels = []
        images = []
        for row in csv_reader:
            if i!=0:
                row = list(map(float, row))
                labels.append(row[0])
                images.append(row[1:])
            i+=1

        labels = np.array(labels, dtype="float")
        images = np.array(images, dtype="float")
        images = np.reshape(images,[len(labels),28,28])
        ### END CODE HERE

        return images, labels

# grader-required-cell

# Test your function
training_images, training_labels = parse_data_from_input(TRAINING_FILE)
validation_images, validation_labels = parse_data_from_input(VALIDATION_FILE)

print(f"Training images has shape: {training_images.shape} and dtype: {training_images.dtype}")
print(f"Training labels has shape: {training_labels.shape} and dtype: {training_labels.dtype}")
print(f"Validation images has shape: {validation_images.shape} and dtype: {validation_images.dtype}")
print(f"Validation labels has shape: {validation_labels.shape} and dtype: {validation_labels.dtype}")


# grader-required-cell

# GRADED FUNCTION: train_val_generators
def train_val_generators(training_images, training_labels, validation_images, validation_labels):
    """
    Creates the training and validation data generators

    Args:
      training_images (array): parsed images from the train CSV file
      training_labels (array): parsed labels from the train CSV file
      validation_images (array): parsed images from the test CSV file
      validation_labels (array): parsed labels from the test CSV file

    Returns:
      train_generator, validation_generator - tuple containing the generators
    """
    ### START CODE HERE

    # In this section you will have to add another dimension to the data
    # So, for example, if your array is (10000, 28, 28)
    # You will need to make it (10000, 28, 28, 1)
    # Hint: np.expand_dims
    training_images = np.reshape(training_images, (len(training_images), 28, 28, 1))
    validation_images = np.reshape(validation_images, (len(validation_images), 28, 28, 1))

    # Instantiate the ImageDataGenerator class
    # Don't forget to normalize pixel values
    # and set arguments to augment the images (if desired)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Pass in the appropriate arguments to the flow method
    train_generator = train_datagen.flow(x=training_images,
                                         y=training_labels,
                                         batch_size=32)

    # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
    # Remember that validation data should not be augmented
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # Pass in the appropriate arguments to the flow method
    validation_generator = validation_datagen.flow(x=validation_images,
                                                   y=validation_labels,
                                                   batch_size=32)

    ### END CODE HERE

    return train_generator, validation_generator

# grader-required-cell

# Test your generators
train_generator, validation_generator = train_val_generators(training_images, training_labels, validation_images, validation_labels)

print(f"Images of training generator have shape: {train_generator.x.shape}")
print(f"Labels of training generator have shape: {train_generator.y.shape}")
print(f"Images of validation generator have shape: {validation_generator.x.shape}")
print(f"Labels of validation generator have shape: {validation_generator.y.shape}")


# grader-required-cell

def create_model():
    ### START CODE HERE

    # Define the model
    # Use no more than 2 Conv2D and 2 MaxPooling2D
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(1024, activation='relu'),

        # tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(25, activation='softmax')
    ])

    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    ### END CODE HERE

    return model

# Save your model
model = create_model()

# Train your model
history = model.fit(train_generator,
                    epochs=15,
                    validation_data=validation_generator)


# Epoch 1/15
# 858/858 [==============================] - 13s 14ms/step - loss: 1.3080 - accuracy: 0.5846 - val_loss: 0.5182 - val_accuracy: 0.8338
# Epoch 2/15
# 858/858 [==============================] - 12s 13ms/step - loss: 0.2602 - accuracy: 0.9119 - val_loss: 0.2090 - val_accuracy: 0.9207
# Epoch 3/15
# 858/858 [==============================] - 14s 17ms/step - loss: 0.1225 - accuracy: 0.9577 - val_loss: 0.1720 - val_accuracy: 0.9555
# Epoch 4/15
# 858/858 [==============================] - 12s 14ms/step - loss: 0.0763 - accuracy: 0.9751 - val_loss: 0.1200 - val_accuracy: 0.9543
# Epoch 5/15
# 858/858 [==============================] - 12s 14ms/step - loss: 0.0534 - accuracy: 0.9817 - val_loss: 0.1811 - val_accuracy: 0.9580
# Epoch 6/15
# 858/858 [==============================] - 11s 13ms/step - loss: 0.0419 - accuracy: 0.9865 - val_loss: 0.1555 - val_accuracy: 0.9625
# Epoch 7/15
# 858/858 [==============================] - 13s 15ms/step - loss: 0.0350 - accuracy: 0.9887 - val_loss: 0.1903 - val_accuracy: 0.9533
# Epoch 8/15
# 858/858 [==============================] - 13s 15ms/step - loss: 0.0301 - accuracy: 0.9900 - val_loss: 0.1146 - val_accuracy: 0.9612
# Epoch 9/15
# 858/858 [==============================] - 12s 14ms/step - loss: 0.0252 - accuracy: 0.9915 - val_loss: 0.2133 - val_accuracy: 0.9502
# Epoch 10/15
# 858/858 [==============================] - 12s 15ms/step - loss: 0.0261 - accuracy: 0.9916 - val_loss: 0.2402 - val_accuracy: 0.9439
# Epoch 11/15
# 858/858 [==============================] - 13s 15ms/step - loss: 0.0216 - accuracy: 0.9928 - val_loss: 0.1492 - val_accuracy: 0.9647
# Epoch 12/15
# 858/858 [==============================] - 13s 15ms/step - loss: 0.0186 - accuracy: 0.9945 - val_loss: 0.2213 - val_accuracy: 0.9589
# Epoch 13/15
# 858/858 [==============================] - 14s 17ms/step - loss: 0.0198 - accuracy: 0.9939 - val_loss: 0.2757 - val_accuracy: 0.9555
# Epoch 14/15
# 858/858 [==============================] - 12s 14ms/step - loss: 0.0173 - accuracy: 0.9948 - val_loss: 0.1734 - val_accuracy: 0.9626
# Epoch 15/15
# 858/858 [==============================] - 12s 14ms/step - loss: 0.0152 - accuracy: 0.9957 - val_loss: 0.1953 - val_accuracy: 0.9619

