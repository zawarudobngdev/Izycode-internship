import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display

import pandas as pd
import numpy as np

from PIL import Image
from torchvision.datasets import ImageFolder
from resizeimage import resizeimage

from skimage.feature import hog
from skimage.color import rgb2gray

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc, accuracy_score

# importing images using torchvision
dataset = ImageFolder("../input/cat-and-dog/training_set/training_set/")

# creating labels dataframe
imgs, labels = zip(*dataset.imgs)
imgs = list(imgs)
labels = list(labels)
labels_df = pd.DataFrame({"image": imgs, "label": labels})
labels_df


# exploring images of dataset
def get_image(path):
    img = Image.open(path)
    return np.array(img)


# showing a dog image
dog_row = labels_df[labels_df.label == 1].reset_index().image[23]
plt.imshow(get_image(dog_row))
plt.show()

# showing a cat image
cat_row = labels_df[labels_df.label == 0].reset_index().image[79]
plt.imshow(get_image(cat_row))
plt.show()


# create image features and flatten into a single row
def create_features(path):
    img = Image.open(path)
    img = resizeimage.resize_cover(img, [56, 56])
    img_arr = np.array(img)
    # flatten three channel color image
    color_features = img_arr.flatten()
    # convert image to greyscale
    grey_image = rgb2gray(img_arr)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm="L2-Hys", pixels_per_cell=(8, 8))
    # combine color and hog features into a single array
    flat_features = np.hstack((color_features, hog_features))
    return flat_features


dog_features = create_features(dog_row)
print(dog_features.shape)


# loop over images to preprocess
def create_feature_matrix(label_df):
    features_list = []

    for img_path in labels_df.image:
        # get features for image
        img_features = create_features(img_path)
        features_list.append(img_features)

    feature_matrix = np.array(features_list)
    return feature_matrix


feature_matrix = create_feature_matrix(labels_df)

# scale feature matrix
# get shape of feature matrix
print("Feature matrix shape is: ", feature_matrix.shape)

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
imgs_stand = ss.fit_transform(feature_matrix)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    imgs_stand, labels_df.label.values, test_size=0.3, random_state=1234123
)

# look at the distrubution of labels in the train set
pd.Series(y_train).value_counts()

# train the model
# define support vector classifier
svm = SVC(kernel="linear", probability=True, random_state=42)

# fit model
svm.fit(X_train, y_train)

# generate predictions
y_pred = svm.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_pred, y_test)
print("Model accuracy is: ", accuracy)

# predict probabilities for X_test using predict_proba
probabilities = svm.predict_proba(X_test)

# select the probabilities for label 1.0
y_proba = probabilities[:, 1]

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(
    y_test, y_proba, pos_label=1
)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title("Receiver Operating Characteristic")
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(
    false_positive_rate, true_positive_rate, label="AUC = {:0.2f}".format(roc_auc)
)

plt.legend(loc=0)
plt.plot([0, 1], [0, 1], ls="--")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")

from random import randint

test = ImageFolder("../input/cat-and-dog/test_set/test_set/")
imgs, labels = zip(*test.imgs)
imgs = list(imgs)
labels = list(labels)

random_ix = randint(0, len(imgs))
label = {0: "Cat", 1: "Dog"}
rand_img = imgs[random_ix]
# create features of the image
test_features = create_features(rand_img)
# predict
prediction = svm.predict([test_features])
print("Prediction: " + label[prediction[0]])
print("Actual: " + label[labels[random_ix]])
# display image
display(Image.open(rand_img))
