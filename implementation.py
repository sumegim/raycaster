import numpy as np
from genevis.render import RaycastRenderer
from genevis.transfer_function import TFColor
from volume.volume import GradientVolume, Volume
from collections.abc import ValuesView
import math


def single_trilinear_interpolation(point: np.ndarray, vertices: np.ndarray):
    """
    Retrieves the interpolated value of a 3D point from the surrounding points.
    :param point: The 3D point for which we want to calculate the linearly interpolated value, with shape (3,)
    :param vertices: Array of all points surrounding the point of interest, with shape (8, 3)
        8 points of a cube and 3 dimensions (x, y, z, value) for each vertex.
    :return: Interpolated value
    """
    # Get vertex to start all calculations
    base_vertex = vertices[0]
    other_vertices = vertices[1:]

    # Boolean arrays that indicates vertices on same axis
    same_x = other_vertices[:, 0] == base_vertex[0]
    same_y = other_vertices[:, 1] == base_vertex[1]
    same_z = other_vertices[:, 2] == base_vertex[2]

    # Vertices for alpha, beta & gamma computation
    alpha_vertex = other_vertices[~same_x &  same_y &  same_z].flatten()
    beta_vertex =  other_vertices[ same_x & ~same_y &  same_z].flatten()
    gamma_vertex = other_vertices[ same_x &  same_y & ~same_z].flatten()

    # Alpha, beta & gamma
    alpha = (point[0] - base_vertex[0]) / (alpha_vertex[0] - base_vertex[0])
    beta = (point[1] - base_vertex[1]) / (beta_vertex[1] - base_vertex[1])
    gamma = (point[2] - base_vertex[2]) / (gamma_vertex[2] - base_vertex[2])

    # Other vertices needed, opposite to vertices identified before
    opp_alpha_vertex = other_vertices[ same_x & ~same_y & ~same_z].flatten()
    opp_beta_vertex =  other_vertices[~same_x &  same_y & ~same_z].flatten()
    opp_gamma_vertex = other_vertices[~same_x & ~same_y &  same_z].flatten()
    opp_base_vertex =  other_vertices[~same_x & ~same_y & ~same_z].flatten()

    final_value = base_vertex[3]      * (1 - alpha) * (1 - beta) * (1 - gamma) + \
                  alpha_vertex[3]     * (    alpha) * (1 - beta) * (1 - gamma) + \
                  beta_vertex[3]      * (1 - alpha) * (    beta) * (1 - gamma) + \
                  gamma_vertex[3]     * (1 - alpha) * (1 - beta) * (    gamma) + \
                  opp_base_vertex[3]  * (    alpha) * (    beta) * (    gamma) + \
                  opp_alpha_vertex[3] * (1 - alpha) * (    beta) * (    gamma) + \
                  opp_beta_vertex[3]  * (    alpha) * (1 - beta) * (    gamma) + \
                  opp_gamma_vertex[3] * (    alpha) * (    beta) * (1 - gamma)

    return final_value
    

def get_voxel(volume: Volume, x: float, y: float, z: float):
    """
    Retrieves the value of a voxel for the given coordinates.
    :param volume: Volume from which the voxel will be retrieved.
    :param x: X coordinate of the voxel
    :param y: Y coordinate of the voxel
    :param z: Z coordinate of the voxel
    :return: Voxel value
    """
    if x < 0 or y < 0 or z < 0 or x >= volume.dim_x or y >= volume.dim_y or z >= volume.dim_z:
        return 0

    x = int(math.floor(x))
    y = int(math.floor(y))
    z = int(math.floor(z))

    return volume.data[x, y, z]

def get_z_voxels(volume: Volume, x: float, y: float):
    """
    Retrieves the array of voxel values for the given coordinates x and y.
    :param volume: Volume from which the array of voxel values will be retrieved
    :param x: X coordinate of the voxel
    :param y: Y coordinate of the voxel
    :return: array of voxel values
    """
    if x < 0 or y < 0 or x >= volume.dim_x or y >= volume.dim_y:
        return 0

    x = int(math.floor(x))
    y = int(math.floor(y))

    return volume.data[x, y]


class RaycastRendererImplementation(RaycastRenderer):
    """
    Class to be implemented.
    """

    def clear_image(self):
        """Clears the image data"""
        self.image.fill(0)

    # TODO: Implement trilinear interpolation

    def render_slicer(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                # Get the voxel coordinate X
                voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                     volume_center[0]

                # Get the voxel coordinate Y
                voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                     volume_center[1]

                # Get the voxel coordinate Z
                voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                     volume_center[2]

                # Get voxel value
                value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)

                # Normalize value to be between 0 and 1
                red = value / volume_maximum
                green = red
                blue = red
                alpha = 1.0 if red > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    # TODO: Implement MIP function
    def render_mip(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                # Get the voxel coordinate X
                voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                     volume_center[0]

                # Get the voxel coordinate Y
                voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                     volume_center[1]

                # Let's try getting the vector of Zs out of Volume.data with coordinates x and y
                voxel_array = get_z_voxels(volume, voxel_coordinate_x, voxel_coordinate_y);
                value = np.max(voxel_array)

                # Normalize value to be between 0 and 1
                red = value / volume_maximum
                green = red
                blue = red
                alpha = 1.0 if red > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    # TODO: Implement Compositing function. TFColor is already imported. self.tfunc is the current transfer function.
    def render_compositing(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        pass

    # TODO: Implement function to render multiple energy volumes and annotation volume as a silhouette.
    def render_mouse_brain(self, view_matrix: np.ndarray, annotation_volume: Volume, energy_volumes: dict,
                           image_size: int, image: np.ndarray):
        # TODO: Implement your code considering these volumes (annotation_volume, and energy_volumes)
        pass


class GradientVolumeImpl(GradientVolume):
    # TODO: Implement gradient compute function. See parent class to check available attributes.
    def compute(self):
        pass
