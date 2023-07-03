# Auxiliar function for turning all numbers *except one* to 0 in an array 
def replace_with_zero(arr, num):
    import numpy as np
    label_mask = np.zeros(arr.shape, dtype = np.uint8)
    label_mask[arr == num] = num
    return label_mask

# The function inputs the path for the mask file, and returns a polygons list with annotations for Yolov8
def annotation_txt_yolo(mask_file):
    # Import necessary libraries
    import cv2 
    import matplotlib.pyplot as plt
    import numpy as np
    
    #mask = cv2.imread('tmp/masks/' + mask_file, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    
    # Which labels (encoded with a number between 1 and 255) appear in the image?
    labels_encoded= np.unique(mask)[1:]
    
    # Create 'polygons': a list of lists. 
    # Every sublist consists of a 'label' and 'point coordinates'
    polygons = []

    for i, label in enumerate(labels_encoded):
        label_mask = replace_with_zero(mask, labels_encoded[i])
        
        # Find the contours of label_mask
        H, W = mask.shape
        contours, hierarchy = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert the contours to polygons
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                polygon = [i]
                for point in cnt:
                    x, y = point[0]
                    polygon.append(round(x / W,3))
                    polygon.append(round(y / H, 3))
                polygons.append(polygon)
                
    return polygons

# This function inputs a directory with a bunch of masks, and writes annotations in yolo format in *the same directory*
def create_yolo_annotations(masks_dir):
    import os
    masks_files = [file for file in os.listdir(masks_dir) if file.endswith((".jpg", ".jpeg", ".png", ".gif"))]
    for mask_file in masks_files:
        polygons = annotation_txt_yolo(os.path.join(masks_dir,mask_file))
        # Write into a txt file, ready for YOLO action!
        with open('{}.txt'.format(os.path.splitext(os.path.join(masks_dir,mask_file))[0]), 'w') as f: 
            for polygon in polygons:
                for i, p in enumerate(polygon):
                    if i == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    else:
                        f.write('{} '.format(p))
    return