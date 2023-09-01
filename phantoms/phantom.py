import numpy as np
import torch
import matplotlib.pyplot as plt
from numba import jit

def get_random_configuration(center_area=0.1, size_low=0, size_high=1, shapes=["rectangle", "ellipse"]):
    
    # center of the rectangle
    x_center = np.random.uniform(-center_area, center_area)
    y_center = np.random.uniform(-center_area, center_area)

    # size of the rectangle
    x_size = np.random.uniform(size_low, size_high)
    y_size = np.random.uniform(size_low, size_high)

    # rotation of rectangle
    rotation_angle = np.random.uniform(0, 180)

    # Shape of object
    shape = np.random.choice(shapes)

    # grey level of rectangle
    graylevel = np.random.uniform(0.2, 0.8)
    
    return x_center, y_center, x_size, y_size, rotation_angle, shape, graylevel

@jit(nopython=True)
def rotation(vector, alpha):
    cosd = np.cos( np.deg2rad(alpha) )
    sind = np.sin( np.deg2rad(alpha) )
    rotation_mat = np.array([[cosd, -sind],
                             [sind,  cosd]] )
    vec_rotation = rotation_mat @ vector
    return vec_rotation

@jit(nopython=True)
def calc_pixel_center_rotation(pixel_size, x_center, y_center, rotation_angle, ii, jj):
    # Calculate pixel center
    x = -1 + pixel_size * jj + pixel_size / 2
    y = -1 + pixel_size * ii + pixel_size / 2
    vector = np.array([x, y])

    # Calculate rotated vector
    v_rot = rotation(vector - np.array([x_center, y_center]) , -rotation_angle) + np.array([x_center, y_center])
    return vector, v_rot

@jit(nopython=True)
def pixel_condition_rectangle(image_size, pixel_size, x_center, y_center, x_size, y_size, rotation_angle, graylevel):
    
    x_size_by_2 = x_size / 2
    y_size_by_2 = y_size / 2
    
    # Temporary phantom for rectangle
    phantom_temp = np.zeros((image_size, image_size))
    
    for ii in range(image_size):
        for jj in range(image_size):
            vector, v_rot = calc_pixel_center_rotation(pixel_size, x_center, y_center, rotation_angle, ii, jj)

            if (x_center - x_size_by_2 <= v_rot[0])\
            and (v_rot[0] <= x_center + x_size_by_2)\
            and (y_center - y_size_by_2 <= v_rot[1])\
            and (v_rot[1] <= y_center + y_size_by_2):
                phantom_temp[ii, jj] = graylevel

    return phantom_temp

@jit(nopython=True)
def pixel_condition_ellipse(image_size, pixel_size, x_center, y_center, x_size, y_size, rotation_angle, graylevel):
    
    x_size_by_2 = x_size / 2
    y_size_by_2 = y_size / 2
    
    x_size_by_2_squared = x_size_by_2**2
    y_size_by_2_squared = y_size_by_2**2
    
    # Temporary phantom for rectangle
    phantom_temp = np.zeros((image_size, image_size))
    
    for ii in range(image_size):
        for jj in range(image_size):
            vector, v_rot = calc_pixel_center_rotation(pixel_size, x_center, y_center, rotation_angle, ii, jj)

            if (v_rot[0] - x_center)**2 / x_size_by_2_squared + (v_rot[1] - y_center)**2/ y_size_by_2_squared <= 1:
                    phantom_temp[ii, jj] = graylevel

    return phantom_temp

@jit(nopython=True)
def phantom_shape(image_size, pixel_size, x_center, y_center, x_size, y_size, rotation_angle, graylevel, shape):
    
    if shape == "rectangle":
        phantom_temp = pixel_condition_rectangle(image_size, pixel_size, x_center, y_center, x_size, y_size, rotation_angle, graylevel)
        
    elif shape  == "ellipse":
        phantom_temp = pixel_condition_ellipse(image_size, pixel_size, x_center, y_center, x_size, y_size, rotation_angle, graylevel)
                
    return phantom_temp

def generate_phantom(resolution=6, number_features=20, x_center=0, y_center=0, x_size=1, y_size=2, rotation_angle=45, graylevel=.5, shape="ellipse"):
    """
    Generates a phantom with a basic shape and a number of features.
    The features are randomly...
            placed in the phantom.
            chosen from the following shapes: rectangle, ellipse.
            rotated.
            scaled.
            placed in the phantom.

    
    Parameters
    ----------
    resolution : int, optional
        Resolution of the phantom. The default is 6.
        image size = 2 ** resolution
        pixel size = 2 / (image size)

    number_features : int, optional
        Number of features in the phantom. The default is 20.

    x_center : float, optional
        x coordinate of the center of the basic shape. The default is 0.

    y_center : float, optional
        y coordinate of the center of the basic shape. The default is 0.

    x_size : float, optional
        x size of the basic shape. The default is 1.

    y_size : float, optional
        y size of the basic shape. The default is 2.

    rotation_angle : float, optional
        Rotation angle of the basic shape. The default is 45.

    graylevel : float, optional
        Graylevel of the basic shape. The default is .5.

    shape : string, optional
        Shape of the basic shape. The default is "ellipse".
        "rectangle", "ellipse"

        
    Returns
    -------
    phantom : numpy array
        Phantom with basic shape and features.

    """
    # Calculate image size and pixel size
    image_size = 2 ** resolution
    pixel_size = 2 /  (image_size)

    # Create basic phantom
    x_center, y_center, x_size, y_size, rotation_angle, shape, graylevel = get_random_configuration(center_area=0.1, size_low=0.9, size_high=1.4, shapes=["rectangle", "ellipse"])
    basic_phantom = phantom_shape(image_size, pixel_size, x_center, y_center, x_size, y_size, rotation_angle, graylevel, shape)

    # Initial filter with only ones (all pixels allowed)
    filter = np.ones((image_size, image_size))

    # Initial collection of feature objects (only zeros->no objects)
    feature_objects = np.zeros((image_size, image_size))

    # iterate over the number of features that will be added
    for ii in range(number_features):
        
        # get random configuration
        x_center, y_center, x_size, y_size, rotation_angle, shape, graylevel = get_random_configuration(center_area=0.2, size_low=0.05, size_high=0.1, shapes=["rectangle", "ellipse"])
        
        # create a phantom feature according to that configuration
        phantom_feature = phantom_shape(image_size, pixel_size, x_center, y_center, x_size, y_size, rotation_angle, graylevel, shape)
        # io.imshow(phantom_feature)
        # io.show()   
        # add the current phantom_feature to the feature object in the pixels, that are not already taken (where filter is 1)
        feature_objects += np.multiply(filter, phantom_feature)  
        
        # get new filter based on new feature_objects.
        # (False * -1) + 1 = 1
        # (True * -1) + 1 = 0
        # So this new filter is 0 precisely in these pixels which are already taken.
        filter = ((feature_objects > 0) * -1) + 1

    # Phantom filter
    phantom_filter = (basic_phantom > 0)

    # phantom
    phantom = basic_phantom + np.multiply(phantom_filter, feature_objects)

    # normalize values
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min())

    return phantom[None, :, :]

def phantom_to_torch(img_numpy):
    """
    Converts a numpy image to a PyTorch tensor and adjusts channels.
    """
    return torch.tensor(img_numpy, dtype=torch.float32).unsqueeze(0)