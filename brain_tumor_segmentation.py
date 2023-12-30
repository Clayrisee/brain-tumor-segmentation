import argparse
import numpy as np
import cv2
import pydicom


IMG_MIMETYPES = [".bmp", ".dcm"]


def read_dcm_images(img_path: str) -> np.array:
    dcm_file = pydicom.read_file(img_path)
    pixel_array = dcm_file.pixel_array
    return cv2.imdecode(pixel_array, cv2.IMREAD_GRAYSCALE)


def load_dir_images(dir: str):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Brain Tumor Segmentation")
    parser.add_argument("-d", "--directory", required=True, type=str,
                        help="path to input directory images")
    args = parser.parse_args()
    return args




