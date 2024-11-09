import cv2
import numpy as np
import os
import random
import scipy
import math
import shutil
from tensorflow.keras.models import load_model
import sys
import io

# Change the default encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')




class NoiseRemover():
  def _erase_circles(img, circles):
      if circles is not None:
          circles = np.uint16(np.around(circles))  # Convert to integers
          for circle in circles[0, :]:
              x = circle[0]  # x coordinate of the circle's center
              y = circle[1]  # y coordinate of the circle's center
              r = circle[2]  # radius of the circle
              # Draw/erase the circle
              img = cv2.circle(img, center=(x, y), radius=r, color=(255), thickness=2)  # erase circle by making it white
      return img

  def _detect_and_remove_circles(img):
      hough_circle_locations = cv2.HoughCircles(img, method = cv2.HOUGH_GRADIENT, dp = 1, minDist = 1, param1 = 50, param2 = 5, minRadius = 0, maxRadius = 2) # We detect unwanted circle by reducing the size and thickness of it so which makes it easier to find with a few parameters!
      if hough_circle_locations is not None:
          img = NoiseRemover._erase_circles(img, hough_circle_locations)
      return img

  def remove_all_noise(img):
      # run some basic tests to get rid of easy-to-remove noise -- first pass
      img = ~img # white letters, black background"
      img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations = 1) # weaken circle noise and line noise
      img = ~img # black letters, white background
      img = scipy.ndimage.median_filter(img, (5, 1)) # remove line noise
      img = scipy.ndimage.median_filter(img, (1, 3)) # weaken circle noise
      img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations = 1) # dilate image to initial stage (erode works similar to dilate because we thresholded the image the opposite way)
      img = scipy.ndimage.median_filter(img, (3, 3)) # remove any final 'weak' noise that might be present (line or circle)

      # detect any remaining circle noise
      img = NoiseRemover._detect_and_remove_circles(img) # after dilation, if concrete circles exist, use hough transform to remove them

      # eradicate any final noise that wasn't removed previously -- second pass
      img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations = 1) # actually performs erosion
      img = scipy.ndimage.median_filter(img, (5, 1)) # finally completely remove any extra noise that remains
      img = cv2.erode(img, np.ones((3, 3), np.uint8), iterations = 2) # dilate image to make it look like the original
      img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations = 1) # erode just a bit to polish fine details

      return img

class CharacterSegmenter():
    def find_nonzero_intervals(vec):
        zero_elements = (vec == 0) * 1 # mark zero-elements as 1 and non-zero elements as 0
        nonzero_borders = np.diff(zero_elements) # find diff between each element and its neighbor (-1 and 1 represent borders, 0 represents segment or non-segment element)
        edges, = np.nonzero(nonzero_borders) # NOTE: comma is vital to extract first element from tuple
        edge_vec = [edges+1] # helps maintain zero-indexing properties (not important to discuss)
        if vec[0] != 0: # special case: catch a segment that starts at the beginning of the array without a 0 border
            edge_vec.insert(0, [0]) # index 0 goes at the beginning of the list to remain proper spatial ordering of intervals
        if vec[-1] != 0: # special case: catch a segment that ends at the end of the array without a 0 border
            edge_vec.append([len(vec)]) # goes at the end of the list to remain proper spatial ordering of intervals
        edges = np.concatenate(edge_vec) # generate final edge list containing indices of 0 elements bordering non-zero segments
        interval_pairs = [(edges[i], edges[i+1]) for i in range(0, len(edges)-1, 2)] # pair up start and end indices
        interval_lengths = [pair[1] - pair[0] for pair in interval_pairs]
        return interval_pairs, interval_lengths

    # def squarify_image(img):
    #     # reshape character crop from (height x width) to (height x height) where height > width
    #     img_height, img_width = img.shape
    #     if img_height > img_width: # make the image fatter
    #         padding = (img_height - img_width) / 2 # force crop to go from 76 x crop_width to 76 x 76 so we can train a CNN
    #         img = cv2.copyMakeBorder(img, top = 0, bottom = 0, left = math.floor(padding), right = math.ceil(padding), borderType = cv2.BORDER_CONSTANT, value = 255)
    #     elif img_height < img_width: # make the image skinnier
    #         margin = (img_width - img_height) / 2
    #         begin_column = int(0 + math.floor(margin))
    #         end_column = int(img_width - math.ceil(margin))
    #         img = img[:, begin_column : end_column]
    #     return img
    def squarify_image(img, margin=5):
        # Find bounding box of non-zero pixels (the character)
        coords = cv2.findNonZero(255 - img)  # Invert to find the character
        x, y, w, h = cv2.boundingRect(coords)  # Get bounding box coordinates
        
        # Add margin around the bounding box
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, img.shape[1] - x)
        h = min(h + 2 * margin, img.shape[0] - y)
        
        # Crop the image based on the bounding box with margin
        img_cropped = img[y:y+h, x:x+w]
        
        # Now we will make it square by padding or cropping
        img_height, img_width = img_cropped.shape
        if img_height > img_width:  # Pad width to make it square
            padding = (img_height - img_width) / 2
            img_cropped = cv2.copyMakeBorder(img_cropped, top=0, bottom=0, 
                                            left=math.floor(padding), right=math.ceil(padding), 
                                            borderType=cv2.BORDER_CONSTANT, value=255)
        elif img_height < img_width:  # Crop width to make it square
            margin = (img_width - img_height) / 2
            begin_column = int(0 + math.floor(margin))
            end_column = int(img_width - math.ceil(margin))
            img_cropped = img_cropped[:, begin_column:end_column]
        
        return img_cropped


    def get_components(img):
        # find number of components
        img = ~img # This inverse the values inside it 0 to 1 and vice versa
        _, markers_original = cv2.connectedComponents(img) # This finds the overlay images as the name suggest
      
        placeholder = ~img

        # perform watershed segmentation
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # The img is converted to RGB adding more channels why cause watershed accepts only a RGB component
        markers = cv2.watershed(~img, markers_original) # gets a unique integer label (0 for the background, 1, 2, ... for each foreground object). 

        markers[placeholder == 255] = -1
        unique_markers = np.unique(markers)

        masks = []
        mask_sizes = []
        mask_start_indices = []
        mask_char_pixels_arrs = []
        if len(unique_markers) > 1:
            for marker in unique_markers[1:]:
                # extract the mask
                mask = np.array((markers != marker) * 255, np.uint8)
                image_height = mask.shape[0]
                num_white_pixels = (mask.sum(axis = 0) / 255) # count number of 255-valued (white) pixels in each column
                num_char_pixels = image_height - num_white_pixels
                mask_start_index = np.nonzero(num_char_pixels > 0)[0][0]

                # crop image to only tightly wrap around the character
                interval, _ = CharacterSegmenter.find_nonzero_intervals(num_char_pixels)
                start, end = interval[0]
                mask = mask[:, start:end]

                # count number of pixels corresponding to a character in the image
                char_pixels_arr = np.count_nonzero(mask == 0, axis = 0)
                num_black_pixels = np.sum(char_pixels_arr)

                # meta information about the mask
                masks.append(mask)
                mask_sizes.append(num_black_pixels)
                mask_start_indices.append(mask_start_index)
                mask_char_pixels_arrs.append(char_pixels_arr)
        return (masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs)

    def segment_characters(masks, mask_sizes, mask_start_indices, mask_char_pixel_arrs):
        # prune out characters with too few pixels (they're just noise)
        masks = [masks[i] for i in range(len(masks)) if mask_sizes[i] > 100]
        mask_start_indices = [mask_start_indices[i] for i in range(len(mask_start_indices)) if mask_sizes[i] > 100]
        mask_char_pixel_arrs = [mask_char_pixel_arrs[i] for i in range(len(mask_char_pixel_arrs)) if mask_sizes[i] > 100]
        mask_sizes = [size for size in mask_sizes if size > 100]

        iters = 0
        while len(masks) < 4 and len(masks) > 0 and iters < 10: # while we haven't found 4 intervals representing 4 characters
            largest_mask_index = np.argmax(mask_sizes) # index of longest interval (split up largest interval because it's the most likely one to have more than one character)

            largest_mask = masks[largest_mask_index]
            largest_mask_size = mask_sizes[largest_mask_index]
            mask_start_index = mask_start_indices[largest_mask_index] # unwrap interval tuple
            mask_char_pixels = mask_char_pixel_arrs[largest_mask_index]

            # when splitting up an interval that might contain 2 characters, we COULD split it directly down the middle, but that's a naive approach
            # instead, just say the best candidate column index to split the characters is the column with the fewest black pixels that's in the middle of this interval (if you include the edges, those might be labeled as the 'best candidate', when in reality they're just the beginning or end edge of a character, and not at the intersection of the two characters)
            padding_value = 0.49 if largest_mask_size < 2200 else 0.1
            margin_length = int(largest_mask.shape[1] * padding_value) # only consider candidates in the middle (padding_value)% of the interval (to remove noisy results on edges of characters), so remove 25% of the interval to the left and 25% of the interval to the right
            new_interval_start = margin_length # start index in the middle (padding_value)% of this interval
            new_interval_end = largest_mask.shape[1] - margin_length # end index in the middle (padding_value)% of this interval
            divider_offset = np.argmin(mask_char_pixels[new_interval_start : new_interval_end]) # found the best candidate column to split the characters -- call this the offset of the character divider from the true start index of the interval

            # preprocess left sub-mask
            left_start = 0
            left_end = new_interval_start + divider_offset
            left_mask = largest_mask[:, left_start : left_end]
            left_char_pixels = mask_char_pixels[left_start : left_end]
            left_start_index = mask_start_index
            left_mask_size = np.sum(left_char_pixels)

            # preprocess right sub-mask
            right_start = new_interval_start + divider_offset
            right_end = largest_mask.shape[1]
            right_mask = largest_mask[:, right_start : right_end]
            right_char_pixels = mask_char_pixels[right_start : right_end]
            right_start_index = mask_start_index + new_interval_start + divider_offset
            right_mask_size = np.sum(right_char_pixels)

            # replace the 'super-interval' (most likely containing two characters) in the intervals list with the two new sub-intervals
            masks[largest_mask_index] = left_mask
            masks.insert(largest_mask_index + 1, right_mask)
            mask_sizes[largest_mask_index] = left_mask_size
            mask_sizes.insert(largest_mask_index + 1, right_mask_size)
            mask_start_indices[largest_mask_index] = left_start_index
            mask_start_indices.insert(largest_mask_index + 1, right_start_index)
            mask_char_pixel_arrs[largest_mask_index] = left_char_pixels
            mask_char_pixel_arrs.insert(largest_mask_index + 1, right_char_pixels)

            # prune out characters with too few pixels (they're just noise)
            masks = [masks[i] for i in range(len(masks)) if mask_sizes[i] > 100]
            mask_start_indices = [mask_start_indices[i] for i in range(len(mask_start_indices)) if mask_sizes[i] > 100]
            mask_char_pixel_arrs = [mask_char_pixel_arrs[i] for i in range(len(mask_char_pixel_arrs)) if mask_sizes[i] > 100]
            mask_sizes = [size for size in mask_sizes if size > 100]

            iters += 1
        return masks, mask_start_indices
    
# Function to apply random augmentations
def apply_random_augmentation(image,augment_choice):
    # Random rotation
    if augment_choice == 0:
        angle = random.randint(-20, 20)
        image = rotate_image(image, angle)
    
    # Random shift (translation)
    if augment_choice == 1:
        shift_x = random.uniform(-0.2, 0.2) * image.shape[1]
        shift_y = random.uniform(-0.2, 0.2) * image.shape[0]
        image = shift_image(image, shift_x, shift_y)
    
    # Random zoom
    if augment_choice == 2:
        zoom_factor = random.uniform(0.8, 1.2)
        image = zoom_image(image, zoom_factor)
    
    # Random Gaussian blur
    if augment_choice == 3:
        blur_value = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (blur_value, blur_value), 0)
    
    return image

# Helper function to rotate the image
def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

# Helper function to shift the image
def shift_image(image, shift_x, shift_y):
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

# Helper function to zoom the image
def zoom_image(image, zoom_factor):
    height, width = image.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    # Resize and then crop/pad back to original size
    image_zoomed = cv2.resize(image, (new_width, new_height))
    if zoom_factor > 1:
        crop_y = (new_height - height) // 2
        crop_x = (new_width - width) // 2
        return image_zoomed[crop_y:crop_y + height, crop_x:crop_x + width]
    else:
        pad_y = (height - new_height) // 2
        pad_x = (width - new_width) // 2
        return cv2.copyMakeBorder(image_zoomed, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT)




# The function that seperates the image and gives back the single elements
def Prepare_Test_Data(imgPath,count,labels):
  # Read the img
  image = cv2.imread(imgPath)

  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Adjust the threshold value if needed
  _, img = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
  img = NoiseRemover.remove_all_noise(img)

  # This first gets the outer regions of the capcha elements using the watershed algorithm
  masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs = CharacterSegmenter.get_components(img)
  if len(masks) > count:
    return "can't Process the image line 286"

  # This then crops out the image properly
  masks, mask_start_indices = CharacterSegmenter.segment_characters(masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs)
  if len(masks) > count:
    return "Can't Process the image line 291"

  # reorder masks and starting indices in ascending order to align them with the proper character for labeling
  mask_start_indices, indices = zip(*sorted(zip(mask_start_indices, [i for i in range(len(mask_start_indices))]))) # make sure intervals are in left-to-right order so we can segment characters properly eg = [59, 4, 121, 141] and the for loop gives [0,1,2,3] and then it 
  masks = [masks[i] for i in indices]
  upgrade_mask = False # Creating a flag for further use
  if len(masks) == count-1:
    upgrade_mask = True 
    proper_mask = []
    for mask in masks:
      # (height,width)
      if mask.shape[1] >= 37: # so if the width is greater than 36 then there is a overlayed image
        mid = mask.shape[1] // 2
        proper_mask.append(mask[:, :mid])
        proper_mask.append(mask[:, mid:])

      else: # IF it is a normal one it will just append to the new list
        proper_mask.append(mask)

  elif len(masks) <count-1:
    return "Can't Process the Image line 311"

  if upgrade_mask:
      char_infos = proper_mask
  else:
      char_infos = masks

  cropped_img_path = os.path.join("static","Labeled Cropped Image")


  if os.path.exists(cropped_img_path):
    shutil.rmtree(cropped_img_path)
  
  os.makedirs(cropped_img_path)

  # save characters to disk
  for index, char_info in enumerate(char_infos):
    # reshape character crop to 76x76
    char_crop = CharacterSegmenter.squarify_image(char_info)
    char_crop = ~char_crop

    char_crop = cv2.resize(char_crop, (56, 56)) 
    
    char_save_file_path = os.path.join(cropped_img_path,f'{labels[index]}.png')

    # save digit to file so we can train a CNN later
    char_save_path = char_save_file_path
    cv2.imwrite(char_save_path, char_crop)
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
        error = Prepare_Test_Data(filepath,count,labels)
    except Exception as e:
        print("There was a error while calling prepare test data",e)

    
    # loading the model 
    model = load_model('capchaPredictor.h5')


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
        
        return True,"".join(output)
    else:
        return False,error
    


