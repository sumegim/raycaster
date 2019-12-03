import numpy as np
from genevis.render import RaycastRenderer
from genevis.transfer_function import TFColor
from volume.volume import GradientVolume, Volume
from collections.abc import ValuesView
import math


# TODO: Implement trilinear interpolation
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


def get_voxels(volume: Volume, ax, ay, az):
    ax = np.trunc(ax).astype(int)
    ay = np.trunc(ay).astype(int)
    az = np.trunc(az).astype(int)

    result = np.zeros(len(ax))

    for i in range(len(ax)):
        x = ax[i]
        y = ay[i]
        z = az[i]
        if not (x < 0 or y < 0 or z < 0 or x >= volume.dim_x or y >= volume.dim_y or z >= volume.dim_z):
            result[i] = volume.data[x, y, z]
    return result


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
                c_i = self.tfunc.get_color(get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z))
                # Normalize value to be between 0 and 1
                red = c_i.r
                green = c_i.g
                blue = c_i.b
                alpha = 1.0

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
                value_max = 0

                vec_k = np.arange(-100, 100, 10)
                x_k = vec_k * view_vector[0]
                y_k = vec_k * view_vector[1]
                z_k = vec_k * view_vector[2]

                vc_base_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + volume_center[0]
                vc_base_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + volume_center[1]
                vc_base_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + volume_center[2]

                vc_vec_x = vc_base_x + x_k
                vc_vec_y = vc_base_y + y_k
                vc_vec_z = vc_base_z + z_k

                value_vec = get_voxels(volume, vc_vec_x, vc_vec_y, vc_vec_z)
                value_max = np.max(value_vec)

                # Normalize value to be between 0 and 1
                red = value_max / volume_maximum
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

                vec_k = np.arange(100, -100, -10)
                x_k = vec_k * view_vector[0]
                y_k = vec_k * view_vector[1]
                z_k = vec_k * view_vector[2]

                vc_base_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + volume_center[0]
                vc_base_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + volume_center[1]
                vc_base_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + volume_center[2]

                vc_vec_x = vc_base_x + x_k
                vc_vec_y = vc_base_y + y_k
                vc_vec_z = vc_base_z + z_k

                c_prev = TFColor(0, 0, 0, 0)
                for k in range(len(vec_k)):
                    # Get the voxel coordinate X

                    c_i = self.tfunc.get_color(get_voxel(volume, vc_vec_x[k], vc_vec_y[k], vc_vec_z[k]))
                    # Get voxel value

                    # # Normalize value to be between 0 and 1
                    c_prev.r = c_i.r * c_i.a + (1 - c_i.a) * c_prev.r
                    c_prev.g = c_i.g * c_i.a + (1 - c_i.a) * c_prev.g
                    c_prev.b = c_i.b * c_i.a + (1 - c_i.a) * c_prev.b


                # Compute the color value (0...255)
                red = math.floor(c_prev.r * 255) if c_prev.r < 255 else 255
                green = math.floor(c_prev.g * 255) if c_prev.g < 255 else 255
                blue = math.floor(c_prev.b * 255) if c_prev.b < 255 else 255
                alpha = 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha


    # TODO: Implement function to render multiple energy volumes and annotation volume as a silhouette.
    def render_mouse_brain(self, view_matrix: np.ndarray, annotation_volume: Volume, energy_volumes: dict,
                           image_size: int, image: np.ndarray):
        # TODO: Implement your code considering these volumes (annotation_volume, and energy_volumes)
        pass


class GradientVolumeImpl(GradientVolume):
    # TODO: Implement gradient compute function. See parent class to check available attributes.
    def compute(self):
        pass
