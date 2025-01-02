# main.py

import importlib
from utils import params
from utils import dataset
import os
import numpy as np
import train
# import relight
import cv2

def main():
    
    data_ = dataset(params.ACQ_PATH)
    
    target_images = np.array(data_.images)
    lps_spherical = np.array(data_.lps_spherical)

    azimuth_elevation = np.array([(azimuth, elevation) for r, azimuth, elevation in lps_spherical])

    print("target_images shape: ", target_images.shape)
    print("light_directions shape: ", azimuth_elevation.shape)


    if params.TRAINING:
        train.train(light_directions=azimuth_elevation,
                    target_images=target_images)
    # else:
    #     relight.test(lps_cartesian=data_.lps_cartesian,
    #                  image_paths=data_.image_paths,
    #                  model_path=params.RTI_MODEL_PATH,
    #                  image_size=(1025,1024))

if __name__ == "__main__":
    main()