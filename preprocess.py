import cv2

from os import environ, path
from numpy import dstack, array
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# Prevent printing during debugging
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Masked image method
def masked_image(image, mask):
    r = image[:, :, 0] * mask
    g = image[:, :, 1] * mask
    b = image[:, :, 2] * mask
    return dstack([r, g, b])

# Includes image processing methods
def image_processing(input_image):
    img_array = cv2.imread(path.join('INPUTS', input_image))
    
    resize_img = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)
    gray_img = rgb2gray(resize_img)
    thresh = threshold_otsu(gray_img)
    thresh_otsu = gray_img < thresh

    filtered = masked_image(resize_img, thresh_otsu)
    filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
    return lab

# Further processing with CLAHE Histogram Equalization
def enc_input(lab):
    clahe = cv2.createCLAHE(clipLimit = 3, tileGridSize = (8,8))
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))  
    enc_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    test_img_new = cv2.cvtColor(enc_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(test_img_new)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    enc_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enc_img

# Gets the features as array inputs for classification
def get_features(enc_img):
    X = list()
    for features in enc_img:
        X.append(features)
    return array(X).reshape(-1, 224, 224, 3)

# Main preprocessing method
def preprocessing(input_image):
    lab = image_processing(input_image)
    enc_img = enc_input(lab)
    features = get_features(enc_img)
    return features