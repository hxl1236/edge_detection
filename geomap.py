import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from xtfEdgeDetect import read_xtf
import os
import matplotlib.animation as animation

def plot_geographical_map(latitude, longitude):
    ''' 
    Plot a geographical map with points.
    '''
    # Convert latitude and longitude to Point geometry
    geometry = [Point(lon, lat) for lat, lon in zip(latitude, longitude)]
    gdf = gpd.GeoDataFrame(geometry, columns=['geometry'])
    
    # Plot the geographical map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the world map
    world.plot(ax=ax, color='lightgrey')
    
    # Plot the points
    gdf.plot(ax=ax, color='red', markersize=5)
    
    # Zoom to the bounding box of the points
    ax.set_xlim(gdf.total_bounds[0], gdf.total_bounds[2] )
    ax.set_ylim(gdf.total_bounds[1], gdf.total_bounds[3] )
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographical Map with Points')
    plt.show()

def plot_geographical_map_animate(latitude, longitude):
    ''' 
    Animate the geographical map with points.
    '''
    # Create initial empty plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geographical Map with Points')
    
    # Load world map data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, color='lightgrey')

    # Initialize empty scatter plot for points
    scatter = ax.scatter([], [], color='red', s=5)

    # Function to update plot with new latitude and longitude
    def update_plot(i):
        # Update scatter plot data
        scatter.set_offsets(np.column_stack((longitude[:i], latitude[:i])))

        # Set plot limits based on current data
        ax.set_xlim(min(longitude), max(longitude))
        ax.set_ylim(min(latitude), max(latitude))

    # Animate the plot
    ani = animation.FuncAnimation(fig, update_plot, frames=len(latitude), repeat=False, interval=100)
    plt.show()

# # Directory containing XTF files
# directory = "../palau_files/palau_files/"
# all_latitude = []
# all_longitude = []
# # all_pitch = []
# # all_roll = []
# # all_heading = []
# # all_depth = []

# # Loop over all XTF files in the directory
# for filename in os.listdir(directory):
#     if filename.endswith(".xtf"):
#         # Read XTF file
#         path = os.path.join(directory, filename)
#         _, longitude, latitude, _, _, _, _ = read_xtf(path)
#         all_latitude.extend(latitude)
#         all_longitude.extend(longitude)

# plot_geographical_map(all_latitude, all_longitude)
#plot_geographical_map_animate(all_latitude, all_longitude)
