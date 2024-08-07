import os
import cv2
import numpy as np
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image

def luminance(color):
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

def find_crop_boundaries(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, x + w, y + h

def calculate_dot_spacing(image_area, desired_drone_count):
    return int(np.sqrt(image_area / desired_drone_count))

def count_drones(upscaled_image, dot_radius, dot_spacing):
    height, width = upscaled_image.shape[:2]
    x_coords, y_coords = np.meshgrid(
        np.arange(dot_radius, width, dot_spacing + 2 * dot_radius),
        np.arange(dot_radius, height, dot_spacing + 2 * dot_radius)
    )
    positions = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    pixel_colors = upscaled_image[positions[:, 1], positions[:, 0]]
    valid_positions = pixel_colors[:, 3] > 0  # Check alpha channel for non-transparent pixels
    return positions[valid_positions]

def process_edges_version(image_path, num_drones):
    # Load the image in grayscale for processing and in color for RGB extraction
    A = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    A_rgb = cv2.imread(image_path)  # For RGB extraction

    # Apply Canny edge detection with adjusted thresholds
    edges = cv2.Canny(A, 30, 100)  # Adjusted thresholds for more detailed edges

    # Get the dimensions of the image
    x, y = edges.shape

    # Find the indices of edge pixels
    edge_indices = np.argwhere(edges > 0)

    # Ensure valid sampling if the number of edge pixels is less than num_drones
    if len(edge_indices) < num_drones:
        num_drones = len(edge_indices)

    # Use KMeans clustering to sample points that are spread out
    kmeans = KMeans(n_clusters=num_drones, random_state=0).fit(edge_indices)
    sampled_indices = kmeans.cluster_centers_.astype(int)

    # Initialize a matrix to store RGB values
    RGB_values = np.zeros((sampled_indices.shape[0], 3))

    # Extract RGB values from the original image
    for i in range(sampled_indices.shape[0]):
        x_coord, y_coord = sampled_indices[i]
        if 0 <= x_coord < A_rgb.shape[0] and 0 <= y_coord < A_rgb.shape[1]:
            RGB_values[i, :] = A_rgb[x_coord, y_coord, :]
        else:
            print(f'Skipping invalid coordinate: {x_coord}, {y_coord}')

    # Concatenate x, y coordinates with RGB values
    output_data = np.column_stack((sampled_indices[:, 1], sampled_indices[:, 0], RGB_values))

    # Define the filename for export
    export_filename = 'output_edges.csv'

    # Export the coordinates and RGB values to a CSV file
    np.savetxt(export_filename, output_data, delimiter=',', fmt='%.6f')
    print(f'Coordinates and RGB values exported to {export_filename}')

    # Initialize a blank image to visualize the selected points
    C = np.zeros((x, y))

    # Mark the selected points in the blank image
    for i in range(sampled_indices.shape[0]):
        qwe = int(sampled_indices[i, 0])
        ewq = int(sampled_indices[i, 1])
        if 0 <= qwe < x and 0 <= ewq < y:
            C[qwe, ewq] = 1
        else:
            print(f"Skipping invalid visualization coordinate: {qwe}, {ewq}")

    # Save the image with marked points
    plt.imshow(C, cmap='gray')
    output_image_path = os.path.splitext(image_path)[0] + '_edges_output.png'
    plt.savefig(output_image_path)
    print(f'Visualization saved to {output_image_path}')

def upscale_image_and_create_matrix(input_logo_path, dotted_output_image_path, dot_radius, desired_drone_count):
    input_logo_image = cv2.imread(input_logo_path, cv2.IMREAD_UNCHANGED)

    # Crop the image
    crop_boundaries = find_crop_boundaries(input_logo_image)
    cropped_image = input_logo_image[crop_boundaries[1]:crop_boundaries[3], crop_boundaries[0]:crop_boundaries[2]]

    # Calculate upscale factor
    height, width = cropped_image.shape[:2]
    height_goal, width_goal = 720, 1280
    upscale_factor = min(width_goal / width, height_goal / height)
    new_dimensions = (int(width * upscale_factor), int(height * upscale_factor))
    upscaled_image = cv2.resize(cropped_image, new_dimensions, interpolation=cv2.INTER_LINEAR)

    # Calculate initial dot spacing
    image_area = new_dimensions[0] * new_dimensions[1]
    min_dot_spacing = 1
    max_dot_spacing = calculate_dot_spacing(image_area, desired_drone_count)
    
    # Binary search for optimal dot spacing
    while min_dot_spacing <= max_dot_spacing:
        dot_spacing = (min_dot_spacing + max_dot_spacing) // 2
        positions = count_drones(upscaled_image, dot_radius, dot_spacing)
        drone_count = len(positions)
        
        print(f"Testing dot_spacing {dot_spacing}, drone_count is {drone_count}")

        if drone_count == desired_drone_count:
            break
        elif drone_count < desired_drone_count:
            max_dot_spacing = dot_spacing - 1
        else:
            min_dot_spacing = dot_spacing + 1

    # Create dotted image and save CSV
    dotted_image = np.zeros((height_goal, width_goal, 4), dtype=np.uint8)
    csv_filename = f"{os.path.splitext(os.path.basename(input_logo_path))[0]}_drones{desired_drone_count}.csv"
    pixel_colors = upscaled_image[positions[:, 1], positions[:, 0]]

    for pos, color in zip(positions, pixel_colors):
        cv2.circle(dotted_image, tuple(pos), dot_radius, color.tolist(), -1)

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([pos[0], pos[1], color[0], color[1], color[2]])

    print(f"Data has been written to {csv_filename}")
    print(f"Drone Count for Dotted Image: {drone_count}")

    # Save the dotted image
    cv2.imwrite(dotted_output_image_path, dotted_image)
    print(f"Dotted image has been saved to {dotted_output_image_path}")

if __name__ == "__main__":
    # Get user inputs
    image_path = input("Enter the path of the image: ")
    version = input("Enter the version (edges/filled): ").lower()
    num_drones = int(input("Enter the number of drones: "))

    # Determine the output file paths based on the input image path
    output_csv_path = os.path.join(os.path.dirname(image_path), "output.csv")
    output_image_path = os.path.join(os.path.dirname(image_path), "output_image.png")

    # Process the image based on the chosen version
    if version == "edges":
        process_edges_version(image_path, num_drones)
    elif version == "filled":
        upscale_image_and_create_matrix(image_path, output_image_path, dot_radius=4, desired_drone_count=num_drones)
    else:
        print("Invalid version entered. Please enter either 'edges' or 'filled'.")
