# The purpose of this file is to use different CV techniques to read an image of a piano, and extract the keys from
# said image. The best function will be incorporated into the main project.

import cv2 as cv
import numpy as np


# Use Sobel Edge Detection to outline piano keys
def basic_sobel_edge_detector(image):
    piano_img_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sobel_piano_1_0 = cv.Sobel(piano_img_grayscale, cv.CV_8U, 1, 0, 3)
    sobel_piano_0_1 = cv.Sobel(piano_img_grayscale, cv.CV_8U, 0, 1, 3)
    sobel_piano_1_1 = cv.Sobel(piano_img_grayscale, cv.CV_8U, 1, 1, 3)

    cv.imshow("1, 0", sobel_piano_1_0)
    cv.imshow("0, 1", sobel_piano_0_1)
    cv.imshow("1, 1", sobel_piano_1_1)
    total = cv.addWeighted(sobel_piano_1_0, 0.5, sobel_piano_0_1, 0.5, 0)

    contours, hierarchy = cv.findContours(total.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    num_objects = len(contours)
    print("Number of Objects Detected: " + str(num_objects))


def improved_sobel_edge_detector(image):
    # reduce noise using 3 x 3  gaussian blur
    noise_reduction = cv.GaussianBlur(image, (3, 3), 0)

    # convert to Grayscale
    piano_img_grayscale = cv.cvtColor(noise_reduction, cv.COLOR_BGR2GRAY)

    # Calculate derivatives in x and y (1, 0) and (0, 1) respectively
    # Depth of output image. CV_16S is used for greater depth (16-bit signed integers)
    sobel_piano_1_0 = cv.Sobel(piano_img_grayscale, cv.CV_16S, 1, 0, 3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    sobel_piano_0_1 = cv.Sobel(piano_img_grayscale, cv.CV_16S, 0, 1, 3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

    # converts output above back to CV_8U
    abs_grad_x = cv.convertScaleAbs(sobel_piano_1_0)
    abs_grad_y = cv.convertScaleAbs(sobel_piano_0_1)

    # approximate gradient by adding x and y gradients
    total_grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    contours, hierarchy = cv.findContours(total_grad.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    num_objects = len(contours)
    print("Number of Objects Detected: " + str(num_objects))
    cv.imshow('Grad', total_grad)


def canny_edge_detector(image):
    # reduce noise using 3 x 3  gaussian blur
    noise_reduction = cv.GaussianBlur(image, (3, 3), 0)

    # convert to Grayscale
    piano_img_grayscale = cv.cvtColor(noise_reduction, cv.COLOR_BGR2GRAY)

    # Canny Edge Detection
    canny_image = cv.Canny(piano_img_grayscale, 50, 200)

    contours, hierarchy = cv.findContours(canny_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    num_objects = len(contours)
    print("Number of Objects Detected: " + str(num_objects))
    cv.imshow('Canny', canny_image)


def threshold_dilation(image):
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(grayscale, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)

    dilation = cv.dilate(thresh, kernel, iterations=2)

    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    num_objects = len(contours)
    print("Number of Objects Detected: " + str(num_objects))

    cv.imshow('threshold', thresh)
    cv.imshow('dilation', dilation)




piano_img = cv.imread('../image_files/piano.jpg')
# basic_sobel_edge_detector(piano_img)
# improved_sobel_edge_detector(piano_img)
# canny_edge_detector(piano_img)
threshold_dilation(piano_img)


while(True):
    key = cv.waitKey(0)

    if (key & 0xFF == ord('q') or key == 27):
        break