import os
import cv2
import numpy as np
from xtfEdgeDetect import get_mpl_colormap, process_image, read_xtf
import matplotlib.pyplot as plt
from geomap import plot_geographical_map

def group_landmarks(landmarks, threshold):
    """
    Group landmarks based on proximity.

    Parameters:
        landmarks (list): List of landmark coordinates.
        threshold (float): Threshold distance for grouping landmarks.

    Returns:
        list: List of groups of landmarks.
    """
    groups = []
    assigned = [False] * len(landmarks)

    for i, landmark in enumerate(landmarks):
        if not assigned[i]:
            nearby_indices = [j for j, other_landmark in enumerate(landmarks) if not assigned[j] and np.linalg.norm(np.array(landmark) - np.array(other_landmark)) < threshold]
            for index in nearby_indices:
                assigned[index] = True
            groups.append([landmarks[index] for index in nearby_indices])

    return groups

def compute_gaussian_model(group):
    """
    Compute Gaussian model for a group of landmarks.

    Parameters:
        group (list): List of landmark coordinates.

    Returns:
        tuple: Mean and covariance matrix of the group.
    """
    mean = np.mean(group, axis=0)
    covariance = np.cov(np.transpose(group))
    
    return mean, covariance

def correct_trajectory(vehicle_latitude, vehicle_longitude, landmarks, threshold):
    """
    Correct vehicle trajectory using landmark information.

    Parameters:
        vehicle_latitude (numpy.ndarray): Array of vehicle latitude coordinates.
        vehicle_longitude (numpy.ndarray): Array of vehicle longitude coordinates.
        landmarks (list): List of landmark coordinates.
        threshold (float): Threshold distance for grouping landmarks.

    Returns:
        tuple: Corrected latitude and longitude arrays.
    """
    groups = group_landmarks(landmarks, threshold)
    corrected_latitude = np.copy(vehicle_latitude)
    corrected_longitude = np.copy(vehicle_longitude)
    
    for group in groups:
        mean, covariance = compute_gaussian_model(group)
        points = np.vstack((vehicle_latitude, vehicle_longitude)).T
        distances = np.sqrt(np.sum(np.dot((points - mean), np.linalg.inv(covariance)) * (points - mean), axis=1))
        within_threshold = distances < threshold
        corrected_latitude[within_threshold] = mean[0]
        corrected_longitude[within_threshold] = mean[1]
                
    return corrected_latitude, corrected_longitude

# Define parameters
directory = "../palau_files/palau_files/"
cmap = get_mpl_colormap(plt.cm.copper)
min_dark_contour_area_threshold = 200
max_dark_contour_area_threshold = 5000
min_bright_contour_area_threshold = 260
max_bright_contour_area_threshold = 2000
max_neighbor_distance = 10
min_area_threshold = 4500
max_area_threshold = 20000
center_distance_threshold = 100
landmark_proximity_threshold = 0.005

# Initialize lists to store vehicle trajectory and landmarks
vehicle_latitude_all = []
vehicle_longitude_all = []
all_landmarks = []

# Loop over XTF files
for filename in os.listdir(directory):
    if filename.endswith(".xtf"):
        path = os.path.join(directory, filename)
        img2, longitude, latitude, _, heading, _, _ = read_xtf(path)
        final_image, landmarks = process_image(img2, longitude, latitude, heading, cmap, min_dark_contour_area_threshold, max_dark_contour_area_threshold,
                                                min_bright_contour_area_threshold, max_bright_contour_area_threshold, max_neighbor_distance,
                                                min_area_threshold, max_area_threshold, center_distance_threshold)
        cv2.imshow('Final Image', final_image)
        cv2.waitKey(0)
        for i, landmark in enumerate(landmarks):
            print(f"Landmark {i + 1}: Latitude = {landmark[0]}, Longitude = {landmark[1]}")
        vehicle_latitude_all.append(latitude)
        vehicle_longitude_all.append(longitude)
        all_landmarks.extend(landmarks)

cv2.destroyAllWindows()
# Correct vehicle trajectory using landmarks
corrected_latitude, corrected_longitude = correct_trajectory(np.concatenate(vehicle_latitude_all), np.concatenate(vehicle_longitude_all),
                                                             all_landmarks, landmark_proximity_threshold)

# Plot corrected trajectory
plot_geographical_map(corrected_latitude, corrected_longitude)
