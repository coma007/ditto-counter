import argparse
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv

def read_csv(path):
    data_dict = {}
    with open(path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            key = row[0]
            value = int(row[1])
            data_dict[key] = value
    return data_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to the dataset")
    args = parser.parse_args()
    return args.data_path


def read_all_images(path):
    images = [path + image for image in listdir(path) if isfile(join(path, image))]
    return images


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def extract_purple(hsv_image):
    lower_purple = np.array([130, 40, 50]) 
    upper_purple = np.array([180, 200, 220])
    return cv2.inRange(hsv_image, lower_purple, upper_purple) 


def convert_to_bin(image):
    return cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        

def get_morphological_features(image, kernel, iter):
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations = iter)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations = iter)
    dilation = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel, iterations = iter)
    erosion = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel, iterations = iter) 
    return opening, closing, dilation, erosion


def distance_transform(feature):
    return cv2.distanceTransform(feature, cv2.DIST_L2, maskSize=5)        


def extract_foreground(dist_transform, background_feature, kernel, iter, percentage):
    sure_bg = cv2.dilate(background_feature, kernel, iterations=iter)
    _, sure_fg = cv2.threshold(dist_transform, percentage * dist_transform.max(), 255, 0) 
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    return sure_fg, sure_bg, unknown


def watershed(image, sure_fg, unknown):
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv2.watershed(image, markers)
    unique_colours = {x for l in markers for x in l}
    return len(unique_colours) - 2


def label2rgb(markers):
    cmap = plt.get_cmap('jet', len(np.unique(markers)))
    return cmap(markers)


def print_result(title, actual_values, predicted_values):
    print(f"{title}-{actual_values}-{predicted_values}")


def save_result(index, actual_values_array, predicted_values_array, actual_values, predicted_values):
    actual_values_array[index] = actual_values
    predicted_values_array[index] = predicted_values


def calculate_mae(predicted_values, actual_values):
    absolute_differences = np.abs(predicted_values - actual_values)
    return np.mean(absolute_differences)