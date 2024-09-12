import cv2
import os
import re 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pydicom as dicom
from io import BytesIO
import preprocessing as pr

def validate_is_dicom(file_content: bytes) -> bool:
    """Gets a byte type object and checks if the content from position 128 till 132 
    mathes the tag 'DICM' for DICOM files.

    Args:
        file_content (bytes): File to be validated

    Returns:
        bool: True if the content from 128 till 132 mathces 'DICM' tag
    """
    print("Validating if the file has a DICM tag ...")
    return file_content[128:132] == b'DICM'


def save_dicom(file_name: str, file_content: bytes) -> None:
    print("Saving file content as DICOM file ...")
    dataset = dicom.dcmread(BytesIO(file_content))
    
    # remove sensitive information in DICOM metadata as patient Name, ID, age, birth date and sex
    dataset = anonymise_dicom_data(dataset=dataset)
    
    # dicom.filewriter.write_file(file_name, file_content, False)
    dicom.dcmwrite(filename=file_name, dataset=dataset, write_like_original=True)


def anonymise_dicom_data(dataset: dicom.FileDataset) ->  None:
    print("Anonymise DICOM data ...")
    dataset.PatientID = None
    dataset.PatientName = None
    dataset.PatientSex = None
    dataset.PatientAge = None
    dataset.PatientBirthDate = None
    return dataset
    

def read_all_images(image_directory):

    if not os.path.exists(image_directory):
        print("(>_<)  Oops! No such directory (-_-)  \n")
        return
    else: 
        print("reading image files in \"{}\" ...".format(image_directory))
        image_name = ''
        image_ext = 'png'
        images = []
        image_type = ''
        image_data = ''
    
    for filename in os.listdir(image_directory):
        filename_rel_path = os.path.join(image_directory,filename)
        if re.search("\\.|jpg|png|jepg|dcm",filename, re.I):
            name =  filename.split('.')
            image_name = name[0]
            image_ext = name[1]
            image_type = 'Color'
            image_data = cv2.imread(filename_rel_path)
            
        else:
            image_name = filename
            image_ext = 'png'
            ds = dicom.dcmread(filename_rel_path)
            image_type =  ds.PhotometricInterpretation  #usually dicom x-ray is MONOCHROME2
            if ds.pixel_array.any():
                ### convert byte raw image data into uint8 in range [0,255]
                image_data = ds.pixel_array - np.min(ds.pixel_array)
                image_data = image_data / np.max(image_data)
                image_data = (image_data * 255).astype(np.uint8)        
        
        if image_data.any():
            images.append((image_data, image_name, image_ext, image_type))

    return(images)


def open_image_file(filename, image_directory):
    # filename_rel_path = f"uploads\\{filename}"
    filename_rel_path = os.path.join(image_directory, filename)
    print(f"Opening image: {filename_rel_path}")

    if re.search("\\.|jpg|png|jepg|dcm",filename, re.I):
        name =  filename.split('.')
        image_name = name[0]
        image_ext = name[1]
        image_type = 'Color'
        image_data = cv2.imread(filename_rel_path)
    else:
        image_name = filename
        image_ext = 'png'
        ds = dicom.dcmread(filename_rel_path)
        image_type =  ds.PhotometricInterpretation		# usually dicom x-ray is MONOCHROME2
        if ds.pixel_array.any():
            ### convert byte raw image data into uint8 in range [0,255]
            print(f"Converting byte raw data from dicom into uint8")
            image_data = ds.pixel_array - np.min(ds.pixel_array)
            image_data = image_data / np.max(image_data)
            image_data = (image_data * 255).astype(np.uint8)        

    if image_data.any():
        return (image_data, image_name, image_ext, image_type)
    else:
        print(f"Image data is missing")
        raise "Image data is missing"


def process_image(filename, image_directory, results_directory):
    original_image, image_name, image_ext, image_type = open_image_file(filename, image_directory)
    
    print("Processing: {} - {} - {}".format(image_name, original_image.shape, image_type))
    if image_type != "MONOCHROME2":
        grey_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        grey_image = original_image

    original_image_width = grey_image.shape[1]
    original_image_height = grey_image.shape[0]

    if original_image_height > 1500:
        scale_ratio = 0.2
    else:
        scale_ratio = 1

    image_height = int(original_image_height*scale_ratio)
    image_width = int(original_image_width*scale_ratio)

    grey_image = cv2.resize(grey_image, (image_width,image_height), interpolation = cv2.INTER_AREA)
    image = cv2.resize(original_image, (image_width,image_height)  , interpolation = cv2.INTER_AREA)
    print("scale ratio: {} \nnew image size: {}-{}".format(scale_ratio, image_width, image_height))
    cv2.imwrite(os.path.join(results_directory,"{}_00-original_image.{}".format(str(image_name),str(image_ext))), image)


    ### auto image enhancment for improving brightness and contrast - histogram strching
    ### not enough
    clip_hist_percent = 15
    enh_image, alpha, beta = pr.automatic_brightness_and_contrast(grey_image, clip_hist_percent=clip_hist_percent)
    cv2.imwrite(os.path.join(results_directory,"{}_01-enh_{}.{}".format(str(image_name),str(clip_hist_percent),str(image_ext))), enh_image)

    ## adaptive equalization for improving image contrast 
    clip_limit = 3
    tile_size_per = 0.46
    tile_size = (image_width//int(image_width*tile_size_per),image_height//int(image_height*tile_size_per))
    enh1_image = pr.adaptive_equalization(grey_image, clip_limit=clip_limit, tile_size=tile_size)
    cv2.imwrite(os.path.join(results_directory,"{}_02-enh_{}.{}".format(str(image_name),str(tile_size),str(image_ext))), enh1_image)


if __name__ == '__main__':
    pass