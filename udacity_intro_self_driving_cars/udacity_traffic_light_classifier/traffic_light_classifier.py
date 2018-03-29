import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

%matplotlib inline

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

## TODO: Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)
## TODO: Print out 1. The shape of the image and 2. The image's label

# The first image in IMAGE_LIST is displayed below (without information about shape or label)
selected_image = IMAGE_LIST[0][0]
plt.imshow(selected_image)

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32, 32))
    
    return standard_im

## TODO: One hot encode an image label
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):
    
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights
    label_types = ['red', 'yellow', 'green']
    # Create a vector of 0's that is the length of the number of classes (3)
    one_hot_encoded = [0] * len(label_types)

    # Set the index of the class number to 1
    one_hot_encoded[label_types.index(label)] = 1 

    return one_hot_encoded

# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

## TODO: Display a standardized image and its label
image_num = 0

selected_image = STANDARDIZED_LIST[image_num][0]
selected_image_label = STANDARDIZED_LIST[image_num][1]

print(selected_image.shape)
print(selected_image_label)

plt.imshow(selected_image)

# Convert and image to HSV colorspace
# Visualize the individual color channels

image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

def hsv_histograms(rgb_image):
    # Convert to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Create color channel histograms
    h_hist = np.histogram(hsv[:,:,0], bins=32, range=(0, 180))
    s_hist = np.histogram(hsv[:,:,1], bins=32, range=(0, 256))
    v_hist = np.histogram(hsv[:,:,2], bins=32, range=(0, 256))
    
    # Generating bin centers
    bin_edges = h_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

    # Plot a figure with all three histograms
    fig = plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.bar(bin_centers, h_hist[0])
    plt.xlim(0, 180)
    plt.title('H Histogram')
    plt.subplot(132)
    plt.bar(bin_centers, s_hist[0])
    plt.xlim(0, 256)
    plt.title('S Histogram')
    plt.subplot(133)
    plt.bar(bin_centers, v_hist[0])
    plt.xlim(0, 256)
    plt.title('V Histogram')
    
    return h_hist, s_hist, v_hist

night_h_hist, night_s_hist, night_v_hist = hsv_histograms(test_im)

## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
# (Optional) Add more image analysis and create more features
def crop_image(rgb_image):
    crop_horizontal = 6
    crop_vertical = 2
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv_crop = hsv[crop_vertical:-crop_vertical, crop_horizontal:-crop_horizontal]
    return hsv_crop

def get_r_value(rgb_image):
    image_crop = crop_image(rgb_image)
    #hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower_red = np.array([120, 35, 100])
    upper_red = np.array([255, 255, 255])
    
    #mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.inRange(image_crop, lower_red, upper_red)
    masked_image = np.copy(image_crop)
    masked_image[mask == 0] = [0, 0, 0]
    
    r_value = np.sum(masked_image[:, :, :])
    
    return r_value

def get_y_value(rgb_image):
    image_crop = crop_image(rgb_image)
    #hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 35, 120])
    upper_yellow = np.array([70, 255, 255])
    
    #mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.inRange(image_crop, lower_yellow, upper_yellow)
    masked_image = np.copy(image_crop)
    masked_image[mask == 0] = [0, 0, 0]
    
    y_value = np.sum(masked_image[:, :, :])
    
    return y_value

def get_g_value(rgb_image):
    image_crop = crop_image(rgb_image)
    #hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([30, 30, 180])
    upper_green = np.array([160, 255, 255])
    
    #mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.inRange(image_crop, lower_green, upper_green)
    masked_image = np.copy(image_crop)
    masked_image[mask == 0] = [0, 0, 0]
    
    g_value = np.sum(masked_image[:, :, :])
    
    return g_value

def get_v_value(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    
    v_value_sum = np.sum(hsv[:, :, 2])
    area = rgb_image.shape[0] * rgb_image.shape[1]
    v_value = v_value_sum / area
    return v_value
    
def create_feature(rgb_image):
    
    ## TODO: Convert image to HSV color space
    ## TODO: Create and return a feature value and/or vector
    r_value = get_r_value(rgb_image)
    y_value = get_y_value(rgb_image)
    g_value = get_g_value(rgb_image)
    
    feature = [r_value,y_value,g_value]
    
    return feature

def get_brightness(rgb_image):
    crop_horizontal = 6
    crop_vertical = 2
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv_crop = hsv[crop_vertical:-crop_vertical, crop_horizontal:-crop_horizontal]
    s = hsv_crop[:, :, 1]
    v = hsv_crop[:, :, 2]
    sum_s = np.sum(s, axis=1)
    sum_v = np.sum(v, axis=1)
    s_max_index = np.argmax(sum_s)
    v_max_index = np.argmax(sum_v)
    brightness = (sum_s, s_max_index, sum_v, v_max_index)
    return brightness

# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    feature = create_feature(rgb_image)
    r_value = feature[0]
    y_value = feature[1]
    g_value = feature[2]
    brightness = get_brightness(rgb_image)
    
    predicted_label = [0, 0, 0]
    sum_s = brightness[0]
    s_max_index = brightness[1]
    sum_v = brightness[2]
    v_max_index = brightness[3]
    v_length = len(sum_v)
    s_length = len(sum_s)
    

    if r_value > y_value and r_value > g_value:
        predicted_label[0] = 1
    elif y_value> r_value and y_value > g_value:
        predicted_label[1] = 1
    elif g_value > r_value and g_value > y_value:
        predicted_label[2] = 1
    else:
        if 0 <= s_max_index <= (s_length/3.0) or 0 <= v_max_index <= (v_length/3.0):
            predicted_label[0] = 1
        elif (s_length/3.0) < s_max_index < (2.0*s_length/3.0) or (v_length/3.0) < v_max_index < (2.0*v_length/3.0):
            predicted_label[1] = 1
        elif (2.0*s_length/3.0) < s_max_index <= s_length or (2.0*v_length/3.0) < v_max_index <= v_length:
            predicted_label[2] = 1
        else:
            return "can not recognize"
    
    return predicted_label   

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

# Visualize misclassified example(s)
## TODO: Display an image in the `MISCLASSIFIED` list 
## TODO: Print out its predicted label - to see what the image *was* incorrectly classified as
mis_image_num = 0
test_mis_im = MISCLASSIFIED[mis_image_num][0]
test_mis_label = MISCLASSIFIED[mis_image_num][1]

print("r_value is:", get_r_value(test_mis_im))
print("y_value is:", get_y_value(test_mis_im))
print("g_value is:", get_g_value(test_mis_im))
print("v_value is:", get_v_value(test_mis_im))
print("max_index is: ", get_brightness(test_mis_im)[1])
# Convert to HSV
mis_hsv = cv2.cvtColor(test_mis_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_mis_label))

# HSV channels
h_mis = mis_hsv[:,:,0]
s_mis = mis_hsv[:,:,1]
v_mis = mis_hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_mis_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

# Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")

