import cv2
import numpy as np
import os
import random
import scipy
import math
import shutil
from tensorflow.keras.models import load_model
import sys
from matplotlib import pyplot as plt
import io

# Change the default encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')



def preprocess_image(img):
    # Convert to grayscale and invert
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned

def segment_characters(img_bin):
    # Find contours
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes and filter based on size and aspect ratio
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter criteria (adjust these values based on your needs)
        min_area = 50  # Minimum area to consider as valid character
        max_area = 5000  # Maximum area to prevent large artifacts
        aspect_ratio = w / float(h)
        
        if (w * h > min_area and 
            w * h < max_area and
            0.2 < aspect_ratio < 5.0):  # Reasonable character aspect ratios
            boxes.append((x, y, w, h))

    # Sort boxes left to right
    boxes = sorted(boxes, key=lambda b: b[0])

    # Merge boxes that are too close (adjust threshold as needed)
    merged_boxes = []
    i = 0
    while i < len(boxes):
        x, y, w, h = boxes[i]
        
        # If this isn't the last box and the next box is very close
        if i < len(boxes) - 1 and (boxes[i+1][0] - (x + w)) < 5:
            # Merge with next box
            new_x = x
            new_y = min(y, boxes[i+1][1])
            new_w = (boxes[i+1][0] + boxes[i+1][2]) - x
            new_h = max(y + h, boxes[i+1][1] + boxes[i+1][3]) - new_y
            merged_boxes.append((new_x, new_y, new_w, new_h))
            i += 2  # Skip next box
        else:
            merged_boxes.append((x, y, w, h))
            i += 1

    return merged_boxes


def prepare_test_data(image_path, count,labels):
    """
    Process a mathematical expression image, segment characters, and save them.
    
    Args:
        image_path (str): Path to the input image
        labels (list): List of labels for each character (must match number of characters)
        output_dir (str): Directory to save cropped characters
        display (bool): Whether to display the segmented characters
    
    Returns:
        list: List of segmented character images (as numpy arrays)
    """
    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    binary = preprocess_image(img)
    
    # Get character boxes
    char_boxes = segment_characters(binary)
    print(len(char_boxes))

    if len(char_boxes) > count or len(char_boxes) < count:
        return "Can't process the image"

    output_dir = os.path.join("static", "Labeled Cropped Image")

    # Clean and create output folder
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Process each character
    segmented_chars = []
    for i, (x, y, w, h) in enumerate(char_boxes):
        char_img = binary[y:y+h, x:x+w]
        

        # Add padding and resize
        padded = cv2.copyMakeBorder(char_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        resized = cv2.resize(padded, (32, 32), interpolation=cv2.INTER_AREA)
        segmented_chars.append(resized)
        
        # Save with label if provided
        save_path = os.path.join(output_dir, f'{labels[i]}.png')

        cv2.imwrite(save_path, resized)
    
        # Display if needed
    # plt.figure(figsize=(15, 3))
    # for i, char_img in enumerate(segmented_chars):
    #     plt.subplot(1, len(segmented_chars), i+1)
    #     plt.imshow(char_img, cmap='gray')
    #     plt.axis('off')
    #     plt.title(f"Char {i+1}")
    #     plt.show()
        
    return None




def predictImage(filepath,count):
    labels = [f'img{i}' for i in range(count)] 

    characters = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z']


    # Seperating the image first
    try:
        error = prepare_test_data(filepath,count,labels)
    except Exception as e:
        error = e
        print("There was a error while calling prepare test data",e)

    
    # loading the model 
    model = load_model('Grsmodel.h5')

    if error == None:
        output = []
        for filename in labels:
            cropped_img_path = os.path.join("static","Labeled Cropped Image",f"{filename}.png")
            # Getting the binary data   
            img_data = cv2.imread(cropped_img_path)
            # Converting into grayscale
            gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            # Resizing the image
            img = cv2.resize(gray_img, (28, 28))
            # print("The img array")
            # print(img)
            # Normalizing the image
            img = img / 255.0            
            # Expanding the dimensions to include channel
            img = np.expand_dims(img, axis=0)  # Shape becomes (1, 28, 28, 1)

            # ============ Predicting the seperated Image ============ #
            y_pred = model.predict(img)
            char_index = np.argmax(y_pred, axis=-1)[0]  # Extract the value from the batch dimension
            output.append(characters[char_index])

        # print("The output is -->","".join(output))
        return True,"".join(output)
    else:
        return False,error
    

result,err = predictImage(r"C:\Coding Stuff\Capcha saver (Automation)\Capcha Images\1.png",5)
print("The result is ==>",result,err)
