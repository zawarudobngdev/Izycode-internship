from google.colab import files
import os

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

# We need to get all the paths for the images to later load them
imagepaths = []

# Go through all the files and subdirectories inside a folder and save path to images inside list
for root, dirs, files in os.walk("leapGestRecog", topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if path.endswith("png"):  # We want only the images
            imagepaths.append(path)

print(len(imagepaths))  # If > 0, then a PNG image was loaded


# This function is used more for debugging and showing results later. It plots the image into the notebook
def plot_image(path):
    img = cv2.imread(path)  # Reads the image into a numpy.array
    img_cvt = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY
    )  # Converts into the corret colorspace (RGB)
    print(img_cvt.shape)  # Prints the shape of the image just to check
    plt.grid(False)  # Without grid so we can see better
    plt.imshow(img_cvt)  # Shows the image
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image " + path)


plot_image(imagepaths[0])  # We plot the first image from our imagepaths array

X = []  # Image data
y = []  # Labels

# Loops through imagepaths to load images and labels into arrays
for path in imagepaths:
    img = cv2.imread(path)  # Reads image and returns np.array
    img = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY
    )  # Converts into the corret colorspace (GRAY)
    img = cv2.resize(img, (320, 120))  # Reduce image size so training can be faster
    X.append(img)

    # Processing label in image path
    category = path.split("/")[2]
    label = int(
        category.split("_")[0][1]
    )  # We need to convert 10_down to 00_down, or else it crashes
    y.append(label)

# Turn X and y into np.array to speed up train_test_split
X = np.array(X, dtype="uint8")
X = X.reshape(
    len(imagepaths), 120, 320, 1
)  # Needed to reshape so CNN knows it's different images
y = np.array(y)

print("Images loaded: ", len(X))
print("Labels loaded: ", len(y))

print(y[0], imagepaths[0])  # Debugging

ts = 0.3  # Percentage of images that we want to use for testing. The rest is used for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

# Creating model

model = Sequential()
model.add(Conv2D(32, (5, 5), activation="relu", input_shape=(120, 320, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    verbose=2,
    validation_data=(X_test, y_test),
)

# save the model to file
model.save("hand_recognition_model.keras")

# Testing model

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test accuracy: {:2.2f}%".format(test_acc * 100))

predictions = model.predict(X_test)  # Make predictions towards the test set

np.argmax(predictions[0]), y_test[0]  # If same, got it right


# Function to plot images and labels for validation purposes
def validate_9_images(predictions_array, true_label_array, img_array):
    # Array for pretty printing and then figure size
    class_names = [
        "down",
        "palm",
        "l",
        "fist",
        "fist_moved",
        "thumb",
        "index",
        "ok",
        "palm_moved",
        "c",
    ]
    plt.figure(figsize=(15, 5))

    for i in range(1, 10):
        # Just assigning variables
        prediction = predictions_array[i]
        true_label = true_label_array[i]
        img = img_array[i]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Plot in a good way
        plt.subplot(3, 3, i)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(
            prediction
        )  # Get index of the predicted label from prediction

        # Change color of title based on good prediction or not
        if predicted_label == true_label:
            color = "blue"
        else:
            color = "red"

        plt.xlabel(
            "Predicted: {} {:2.0f}% (True: {})".format(
                class_names[predicted_label],
                100 * np.max(prediction),
                class_names[true_label],
            ),
            color=color,
        )
    plt.show()


validate_9_images(predictions, y_test, X_test)

y_pred = np.argmax(
    predictions, axis=1
)  # Transform predictions into 1-D array with label number

# H = Horizontal
# V = Vertical

pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=[
        "Predicted Thumb Down",
        "Predicted Palm (H)",
        "Predicted L",
        "Predicted Fist (H)",
        "Predicted Fist (V)",
        "Predicted Thumbs up",
        "Predicted Index",
        "Predicted OK",
        "Predicted Palm (V)",
        "Predicted C",
    ],
    index=[
        "Actual Thumb Down",
        "Actual Palm (H)",
        "Actual L",
        "Actual Fist (H)",
        "Actual Fist (V)",
        "Actual Thumbs up",
        "Actual Index",
        "Actual OK",
        "Actual Palm (V)",
        "Actual C",
    ],
)
