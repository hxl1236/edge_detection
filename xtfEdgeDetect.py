import cv2
import numpy as np
import pyxtf
import matplotlib.pyplot as plt   
from pixeltogeo import frame

def get_mpl_colormap(cmap_name):
    ''' 
    Create a colormap for plotting
    '''
    # Enables use of matplotlib colormaps, from https://stackoverflow.com/a/52501371
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)

def read_xtf(path):
    ''' 
    Read in the xtf file and return the sonar image, latitude, longitude, heading,
    pitch, roll, and depth.
    '''
    (fh, packets) = pyxtf.xtf_read(path)
    # Get sonar pkts from xtf file
    sonar_packets = packets[pyxtf.XTFHeaderType.sonar]

    data_array = pyxtf.concatenate_channel(sonar_packets, file_header=fh, channel=0, weighted=True)
    data_array2 = pyxtf.concatenate_channel(sonar_packets, file_header=fh, channel=1, weighted=True)

    longitude = []
    latitude = []
    depth = []
    heading = []
    pitch = []
    roll = []

    for ping in sonar_packets:
        # Extract sensor coordinates and heading
        longitude.append(ping.SensorXcoordinate)
        latitude.append(ping.SensorYcoordinate)
        depth.append(ping.SensorDepth)
        heading.append(ping.SensorHeading)
        pitch.append(ping.SensorPitch)
        roll.append(ping.SensorRoll)

    # Scale down the 16-bit values to 8-bit values
    scale_factor = 2
    shift_factor = 0
    scaled_data_array = ((data_array / 65535) * 255 * scale_factor + shift_factor).astype(np.uint8)
    scaled_data_array2 = ((data_array2 / 65535) * 255 * scale_factor + shift_factor).astype(np.uint8)

    # Concatenate port and stbd sonar pings
    concatenated_array = np.hstack((scaled_data_array, scaled_data_array2))# + 20   # Add shift to make image brighter
    img = concatenated_array

    return img, longitude, latitude, depth, heading, pitch, roll

def get_middle_value(arr):
    ''' 
    Obtain the center element of the array
    '''
    n = len(arr)
    if n % 2 == 1:  # Odd number of elements
        return arr[n // 2]
    else:  # Even number of elements
        middle_index1 = n // 2 - 1
        middle_index2 = n // 2
        return (arr[middle_index1] + arr[middle_index2]) / 2
    
def merge_contours(bright_contours, dark_contours, max_neighbor_distance):
    ''' 
    Merge the neighboring contours from bright regions and dark regions (two groups) together.
    Return merged and unmerged contours.
    '''
    merged_bright_contours = []
    unmerged_dark_contours = []
    unmerged_bright_contours = []

    for bright_contour in bright_contours:
        merged_bright = bright_contour
        merged = False
        updated_dark_contours = []  # New list to store dark contours without the contour being merged
        for dark_contour in dark_contours:
            merge_flag = False  # Flag to check if the dark contour is merged
            for point in dark_contour[:, 0]:
                pt = (int(point[0]), int(point[1]))  # Convert coordinates to integers
                dist = cv2.pointPolygonTest(bright_contour, pt, True)
                if dist > 0 and dist < max_neighbor_distance:
                    merged_bright = cv2.convexHull(np.concatenate((merged_bright, dark_contour)))
                    merge_flag = True
                    merged = True
                    break
            if not merge_flag:  # If the contour is not merged, add it to the updated dark contours list
                updated_dark_contours.append(dark_contour)
        dark_contours = updated_dark_contours  # Update dark_contours array after removing the merged contour
        if merged:
            merged_bright_contours.append(merged_bright)
        else:
            unmerged_bright_contours.append(bright_contour)

    unmerged_dark_contours = dark_contours
    return merged_bright_contours, unmerged_bright_contours, unmerged_dark_contours

def merge_contours2(contours, max_merge_distance):
    ''' 
    Merge the neighboring contours from a single group of contours
    '''
    merged_contours = []
    unmerged_contours = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True) # Sort contours by area

    merged_indices = set() # Indices of contours already merged

    for i in range(len(contours)):
        if i in merged_indices:
            continue
        merged_contour = contours[i]
        merged_indices.add(i)  # Add the current contour to merged indices
        merged_centroid = np.mean(merged_contour, axis=0)[0]

        merged_this_round = False

        for j in range(i+1, len(contours)):
            if j in merged_indices:
                continue
            contour = contours[j]
            contour_centroid = np.mean(contour, axis=0)[0]
            distance = np.linalg.norm(merged_centroid - contour_centroid) # Compute distance between centroids
            if distance <= max_merge_distance: # Check distance
                # Merge contours by adding their points together
                merged_contour = np.concatenate((merged_contour, contour))
                merged_indices.add(j)  # Add the current contour to merged indices
                merged_centroid = np.mean(merged_contour, axis=0)[0]
                merged_this_round = True

        if not merged_this_round:
            unmerged_contours.append(merged_contour)

        # Compute the convex hull of the merged contour
        merged_hull = cv2.convexHull(merged_contour)
        merged_contours.append(merged_hull)

    # Append remaining unmerged contours
    for i in range(len(contours)):
        if i not in merged_indices:
            unmerged_contours.append(contours[i])

    return merged_contours, unmerged_contours

def darkness(img, cmap, min_contour_area_threshold, max_contour_area_threshold):
    ''' 
    Detect the edge of the landmarks from dark regions and return the contours of them.
    '''
    alpha = 10
    beta = 30
    high_contrast_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    img_blur = cv2.GaussianBlur(high_contrast_result, (7, 7), 0)

    edges_result = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    contours, _ = cv2.findContours(edges_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(edges_result)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    dilated_mask = cv2.dilate(mask, None, iterations=3)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [contour for contour in contours if min_contour_area_threshold <= cv2.contourArea(contour) <= max_contour_area_threshold]

    merged_contours,unmerged_contours = merge_contours2(filtered_contours, 100) # Merge neighboring contours within a certain distance
    merged_contours = merged_contours + unmerged_contours
    num_landmarks = len(merged_contours)
    print("Number of Dark Landmarks:", num_landmarks)

    img_color = cv2.applyColorMap(img, cmap)
    img_with_dark_contours = img_color.copy()
    cv2.drawContours(img_with_dark_contours, merged_contours, -1, (0, 255, 0), 2)  # Draw green contours
    return img_with_dark_contours, merged_contours

def brightness(img, cmap, min_contour_area_threshold, max_contour_area_threshold):
    ''' 
    Detect the edge of the landmarks from bright regions and return the contours of them.
    '''
    inverted_img_gray = 255 - img
    # Adjust contrast and brightness levels in the inverted grayscale image
    alpha = 2.5
    beta = 30
    high_contrast_result = cv2.convertScaleAbs(inverted_img_gray, alpha=alpha, beta=beta)
    img_blur = cv2.GaussianBlur(high_contrast_result, (1, 1), 0)

    edges_result = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    contours, _ = cv2.findContours(edges_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(edges_result)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Dilate the mask to connect nearby regions
    dilated_mask = cv2.dilate(mask, None, iterations=3)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [contour for contour in contours if min_contour_area_threshold <= cv2.contourArea(contour) <= max_contour_area_threshold]
    num_landmarks = len(filtered_contours)
    print("Number of Bright Landmarks:", num_landmarks)

    # Drawing the contours on the original image
    img_color = cv2.applyColorMap(img, cmap)
    img_with_bright_contours = img_color.copy()
    cv2.drawContours(img_with_bright_contours, filtered_contours, -1, (255, 0, 0), 2)  # Draw blue contours
    return img_with_bright_contours, filtered_contours

def get_contour_center(contour):
    ''' 
    Obtain the pixel coordinates x and y of the given contours.
    '''
    M = cv2.moments(contour)
    if M["m00"] != 0: # total area of contour
        cX = int(M["m10"] / M["m00"]) #sum of the x-coordinates of all pixels/total area of contour
        cY = int(M["m01"] / M["m00"]) #sum of the y-coordinates of all pixels/total area of contour
        return cX, cY
    else:
        return None

def adjust_coordinates(center, image_shape):
    ''' 
    Adjust the pixel coordinates with respect to the center of the image as (0,0)
    '''
    center_x, center_y = center
    image_height, image_width = image_shape[:2]
    adjusted_center_x = center_x - image_width // 2
    adjusted_center_y = image_height // 2 - center_y  # Reversed due to image coordinates
    return adjusted_center_x, adjusted_center_y

def filter_contours_by_distance_from_center(contours, center, distance_threshold):
    ''' 
    Filtered out the contours that are close to the center of the image
    '''
    filtered_contours = []
    for contour in contours:
        contour_center = get_contour_center(contour)
        if contour_center is not None:
            distance = np.linalg.norm(np.array(contour_center) - np.array(center))
            if distance > distance_threshold:
                filtered_contours.append(contour)
    return filtered_contours

def process_image(img2, longitude, latitude, heading, cmap, min_dark_contour_area_threshold, max_dark_contour_area_threshold,
                  min_bright_contour_area_threshold, max_bright_contour_area_threshold, 
                  max_neighbor_distance, min_area_threshold, max_area_threshold, center_distance_threshold):
    ''' 
    Process one sonar image based on its geographical information.
    Return the filtered contours and the geographical coordinates of the landmarks.
    '''
    img2 = cv2.resize(img2, (img2.shape[1] // 3, img2.shape[0] // 3))
    img2_colored = cv2.applyColorMap(img2, cmap)

    _, dark_contours = darkness(img2, cmap, min_dark_contour_area_threshold, max_dark_contour_area_threshold)
    _, bright_contours = brightness(img2, cmap, min_bright_contour_area_threshold, max_bright_contour_area_threshold)

    merged_bright_contours, unmerged_bright_contours, unmerged_dark_contours = merge_contours(bright_contours, dark_contours, max_neighbor_distance)
    all_contours = merged_bright_contours + unmerged_bright_contours + unmerged_dark_contours

    merged_contours, unmerged_contours = merge_contours2(all_contours, 50)
    all_contours2 = merged_contours + unmerged_contours

    filtered_contours = [contour for contour in all_contours2 if cv2.contourArea(contour) >= min_area_threshold and cv2.contourArea(contour) <= max_area_threshold]
    image_center = (img2.shape[1] // 2, img2.shape[0] // 2)
    filtered_contours = filter_contours_by_distance_from_center(filtered_contours, image_center, center_distance_threshold)
    overlaid_img = img2_colored.copy()
    cv2.drawContours(overlaid_img, filtered_contours, -1, (0, 255, 0), 2)  # Draw filtered contours

    landmarks = set()  # Using a set to store unique landmarks

    lat0_value = get_middle_value(latitude)
    lon0_value = get_middle_value(longitude)
    heading_value = get_middle_value(heading)
    pixelIToM_value = 0.1
    pixelJToM_value = 0.1

    frame_instance = frame(lat0_value, lon0_value, heading_value, pixelIToM_value, pixelJToM_value) 
    for contour in filtered_contours:
        center = get_contour_center(contour)
        if center is not None:
            adjusted_center = adjust_coordinates(center, img2.shape)
            pixel_i, pixel_j = adjusted_center
            lat, lon = frame_instance.pixelToGeo(pixel_i, pixel_j)
            landmark = (lat, lon)
            if landmark not in landmarks:  # Check if the landmark is unique
                landmarks.add(landmark)
                cv2.circle(overlaid_img, center, 5, (0, 0, 255), -1)  # Red circle

    return overlaid_img, list(landmarks)  # Convert set back to list before returning



