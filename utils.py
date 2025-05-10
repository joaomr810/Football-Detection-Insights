## ----- All auxiliary functions used in the notebook ---------

import cv2
import json
import numpy as np


def draw_boxes(image, label_file, color):
    """
    Draws bounding boxes on an image/frame using pixel coordinates annotated in a .txt file 
    following the MOT format: <frame, id, x, y, width, height, conf, class, visibility>
 
    Args:
        image (np.ndarray): The image/frame on which to draw boxes
        label_file (str): Path to the .txt file containing the frame detections
        color (tuple): BGR tuple for the cv2.rectangle bounding box color (e.g., (255, 0, 0) for blue)   
    """
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            _, _, x, y, w, h, *_ = map(float, parts)
            top_left = (int(x), int(y))
            bottom_right = (int(x + w), int(y + h))
            cv2.rectangle(image, top_left, bottom_right, color, 2)



def annotate_pitch_corners(image_path='./images/frame.png', scale=0.5, output_image_path='./images/frame_annotated.png', output_json_path='pitch_corners.json'):
    """
    Opens an image and allows the user to manually select points of a football pitch.
    Saves the selected pixel coordinates to a JSON file and the annotated image.

    Args:
        image_path (str): Path to the input image
        scale (int or float): Resizing factor for display purposes
        output_image_path (str): Path to save the annotated image with drawn corners
        output_json_path (str): Path to save the selected coordinates as a JSON file

    Returns:
        corners (list of tuples): List of the selected corner coordinates in original image scale
    """

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    clone = image.copy()                                    
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    corners = []

    
    def mouse_callback(event, x, y, flags, param):
        """
        Mouse callback that registers left-mouse clicks on the image.

        On each left-click, the clicked point is converted to the original image scale, drawn on both 
        the original and resized images, and added to the corners list with the original pixel coordinates.

        Args:
            event: The type of mouse event (e.g. click)
            x (int): x-coordinate of the mouse event
            y (int): y-coordinate of the mouse event
            flags (int): OpenCV event flags (not used here but required in the setMouseCallback function)
            param (numpy.ndarray): The original image on which to draw the annotation
        """

        if event == cv2.EVENT_LBUTTONDOWN:          # The function will only annotate pixels with left-mouse clicks

            # Converting the clicked coordinates from resized image to the original image scale
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            corners.append((orig_x, orig_y))

            # Draw a red circle at the selected point on the original image and write its coordinates next to it
            cv2.circle(param, (orig_x, orig_y), 7, (0, 0, 255), -1)
            cv2.putText(param, f'({orig_x}, {orig_y})', (orig_x + 10, orig_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw a red circle on the resized image when the corners are being chosen and update with it
            cv2.circle(resized, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Select Corners', resized)

    # Show window and set mouse callback function
    cv2.namedWindow('Select Corners', cv2.WINDOW_NORMAL)
    cv2.imshow('Select Corners', resized)
    cv2.setMouseCallback('Select Corners', mouse_callback, clone)

    # Press any key to leave the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save outputs in a .json file
    cv2.imwrite(output_image_path, clone)
    with open(output_json_path, 'w') as f:
        json.dump(corners, f)

    return corners


def image_to_pitch(x, y, H):
    """
    Applies a homography transformation to map a point from image coordinates to pitch (top-down) coordinates.

    Args:
        x (float): x-coordinate in the original image.
        y (float): y-coordinate in the original image.
        H (numpy.ndarray): 3x3 homography matrix.

    Returns:
        tuple: (x', y') coordinates of the point in the transformed pitch perspective.    
    """
    point = np.array([x, y, 1]).reshape((3, 1))         # Converts (x, y) to homogeneous coordinates as a 3x1 column vector
    mapped = H @ point                                  # Applies the homography transformation: maps the point to the new perspective
    mapped /= mapped[2]                                 # Normalizes by dividing by the third coordinate to convert back to 2D
    return float(mapped[0]), float(mapped[1])           # Returns the transformed (x, y) coordinates as floats