import numpy as np
import pandas as pd
from genevis.render import RaycastRenderer
from genevis.transfer_function import TFColor
from volume.volume import GradientVolume, Volume
from collections.abc import ValuesView
import math
from tqdm import tqdm

from itertools import product

# Get the colour specified in structures.csv for each annotation
structures = pd.read_csv('../meta/structures.csv')
structures.index = structures.database_id
colour_table = pd.concat([structures.color.str.slice(0,2), structures.color.str.slice(2,4), structures.color.str.slice(4,6)], axis=1)
colour_table.columns = ["r", "g", "b"]
colour_table = colour_table.apply(lambda col: col.apply(int, base=16), axis=0)
colour_table = colour_table / 255                                                    # Normalize the data
colour_table = colour_table.append(pd.Series([0,0,0], name=0, index=['r','g','b']))  # Add black

def single_trilinear_interpolation(point_raw: np.ndarray, vertices_raw: np.ndarray, view_inverse: np.ndarray) -> float:
    """
    Retrieves the interpolated value of a 3D point from the surrounding points.
    :param point: The 3D point for which we want to calculate the linearly interpolated value, with shape (3,)
    :param vertices_raw: Array of all points surrounding the point of interest, with shape (8, 4)
        8 points of a cube and 3 dimensions (x, y, z) plus the value for each vertex.
    :param view_inverse: The inverse of the transformation matrix provided by the framework
    :return: Interpolated value
    """
    # Transpose everything to the cube coordinate system
    # view_inverse = np.around(view_inverse, 1)
    # point = view_inverse @ point_raw
    # # Should get vertices for each point
    # coords = np.stack([np.floor(point), np.ceil(point)]).T
    # vertices_of_point = np.array(list(product(coords[0], coords[1], coords[2])))
    # values = get_voxels(volume, vertices_of_point.T[0], vertices_of_point.T[1], vertices_of_point.T[2])
    # vertices = np.concatenate([vertices_of_point, values.reshape((-1, 1))], axis=1)

    point = point_raw
    vertices = vertices_raw

    # If point is already on int coords, it's vertices will be the same as the point
    if np.isclose(vertices,vertices[0]).all():
        # Just return the value of any vertex, they are all the same
        return vertices[0][3]
    else: 
    # Do the interpolation

        # Get vertex to start all calculations
        base_vertex = vertices[0]
        other_vertices = vertices[1:]

        # Boolean arrays that indicates vertices on same axis
        same_x = other_vertices[:, 0] == base_vertex[0]
        same_y = other_vertices[:, 1] == base_vertex[1]
        same_z = other_vertices[:, 2] == base_vertex[2]

        # Vertices for alpha, beta & gamma computation
        alpha_vertex = other_vertices[~same_x & same_y & same_z].flatten()
        beta_vertex = other_vertices[same_x & ~same_y & same_z].flatten()
        gamma_vertex = other_vertices[same_x & same_y & ~same_z].flatten()

        # Alpha, beta & gamma
        alpha = (point[0] - base_vertex[0]) / (alpha_vertex[0] - base_vertex[0])
        beta = (point[1] - base_vertex[1]) / (beta_vertex[1] - base_vertex[1])
        gamma = (point[2] - base_vertex[2]) / (gamma_vertex[2] - base_vertex[2])

        # Other vertices needed, opposite to vertices identified before
        opp_alpha_vertex = other_vertices[same_x & ~same_y & ~same_z].flatten()
        opp_beta_vertex = other_vertices[~same_x & same_y & ~same_z].flatten()
        opp_gamma_vertex = other_vertices[~same_x & ~same_y & same_z].flatten()
        opp_base_vertex = other_vertices[~same_x & ~same_y & ~same_z].flatten()

        final_value = base_vertex[3] * (1 - alpha) * (1 - beta) * (1 - gamma) + \
            alpha_vertex[3] * (alpha) * (1 - beta) * (1 - gamma) + \
            beta_vertex[3] * (1 - alpha) * (beta) * (1 - gamma) + \
            gamma_vertex[3] * (1 - alpha) * (1 - beta) * (gamma) + \
            opp_base_vertex[3] * (alpha) * (beta) * (gamma) + \
            opp_alpha_vertex[3] * (1 - alpha) * (beta) * (gamma) + \
            opp_beta_vertex[3] * (alpha) * (1 - beta) * (gamma) + \
            opp_gamma_vertex[3] * (alpha) * (beta) * (1 - gamma)

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

def interpolate(volume: Volume, points_raw: np.ndarray, view_inverse: np.ndarray):
    """ 
    :param points_raw: shape == (n_points, 3)
    """
    # Should get vertices for each point
    # point = points_raw[0]
    # coords = np.stack([np.floor(point), np.ceil(point)]).T
    # vertices_of_point = np.array(list(product(coords[0], coords[1], coords[2])))
    # values = get_voxels(volume, vertices_of_point.T[0], vertices_of_point.T[1], vertices_of_point.T[2])
    # vertices = np.concatenate([vertices_of_point, values.reshape((-1, 1))], axis=1)
    # return vertices

    # Vectorized vertices
    # Could be a bit slow for list comprehension, might change
    # points = (view_inverse @ points_raw.T).T
    points = points_raw

    coords_raw = np.stack([np.floor(points), np.ceil(points)], axis=1)
    condition = np.diff(coords_raw, axis=1).astype(bool)
    broad_condition = np.broadcast_to(condition, coords_raw.shape)
    if broad_condition.all():
        coords = coords_raw
    else:
        coords = np.where(broad_condition, coords_raw, np.array([[coords_raw[~broad_condition][0]], [coords_raw[~broad_condition][1] + 1]]))
    
    vertices_of_point = np.array([list(product(coords[i][:,0], coords[i][:,1], coords[i][:,2])) 
                                  for i in range(coords.shape[0])])
    values = np.array([get_voxels(volume, vertices_of_point[i].T[0], vertices_of_point[i].T[1], vertices_of_point[i].T[2])
                       for i in range(vertices_of_point.shape[0])])
    vertices_raw = [np.concatenate([vertices_of_point[i], values[i].reshape(-1,1)], axis=1) 
                    for i in range(vertices_of_point.shape[0])]
    
    final_values = np.array([single_trilinear_interpolation(points[i], vertices_raw[i], view_inverse) 
                             for i in range(points.shape[0])])
    return final_values


def get_voxels(volume: Volume, xs_raw: np.ndarray, ys_raw: np.ndarray, zs_raw: np.ndarray) -> np.ndarray:
    """
    Return a whole array of voxel values, given arrays of coordinates xs_raw, ys_raw and zs_raw
    """

    xs = np.floor(xs_raw).astype(int)
    ys = np.floor(ys_raw).astype(int)
    zs = np.floor(zs_raw).astype(int)

    invalid_xs = (xs < 0) | (xs >= volume.dim_x)
    invalid_ys = (ys < 0) | (ys >= volume.dim_y)
    invalid_zs = (zs < 0) | (zs >= volume.dim_z)

    test = np.stack([invalid_xs, invalid_ys, invalid_zs]).sum(axis=0)

    xs[invalid_xs] = 0
    ys[invalid_ys] = 0
    zs[invalid_zs] = 0

    result = np.where(test, np.zeros_like(test), volume.data[xs, ys, zs])

    return result


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
                value = get_voxel(volume, voxel_coordinate_x,
                                  voxel_coordinate_y, voxel_coordinate_z)

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

    def vector_render_mip(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Define basis matrix of viewplane coord vectors in volume coord system
        basis_matrix = np.stack(
            [view_matrix[0:3], view_matrix[4:7], view_matrix[8:11]]).T
        view_inverse = np.linalg.inv(basis_matrix)

        image_center = image_size // 2

        volume_center = np.array(
            [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2])
        volume_maximum = volume.get_maximum()

        max_range = max(volume.dim_x, volume.dim_y, volume.dim_y)
        sample_start = -max_range / 2
        sample_end = max_range / 2
        sample_step = 1
        n_samples = math.ceil((sample_end - sample_start) / sample_step)
        view_samples = np.arange(sample_start, sample_end, sample_step)

        step = 2 if self.interactive_mode else 1
        for i in tqdm(range(0, image_size, step)):
            for j in range(0, image_size, step):
                raw_points = np.stack(
                    [np.full(n_samples, i - image_center),
                     np.full(n_samples, j - image_center),
                     view_samples])

                points = (basis_matrix @ raw_points) + \
                    volume_center.reshape(-1, 1)

                # voxels = interpolate(volume, points.T, view_inverse)
                voxels = get_voxels(volume, points[0], points[1], points[2])

                value = voxels.max()
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

    def render_mip(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()
        self.vector_render_mip(view_matrix, volume, image_size, image)

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

                vec_k = np.arange(100, -100, -2)
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
                    vx = get_voxel(volume, vc_vec_x[k], vc_vec_y[k], vc_vec_z[k])
                    if vx < 1:
                        continue

                    c_i = self.tfunc.get_color(vx)
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

    def render_annotation_compositing(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        self.clear_image()
        u_vector = view_matrix[0:3]
        v_vector = view_matrix[4:7]
        view_vector = view_matrix[8:11]
        image_center = image_size / 2
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in tqdm(range(0, image_size, step)):
            for j in range(0, image_size, step):

                vec_k = np.arange(100, -100, -2)
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
                default_gamma = 0.5

                for k in range(len(vec_k)):

                    vx = get_voxel(volume, vc_vec_x[k], vc_vec_y[k], vc_vec_z[k])
                    if vx < 1:
                        continue

                    c_i = colour_table.loc[vx]

                    c_prev.r = c_i.r * default_gamma + (1 - default_gamma) * c_prev.r
                    c_prev.g = c_i.g * default_gamma + (1 - default_gamma) * c_prev.g
                    c_prev.b = c_i.b * default_gamma + (1 - default_gamma) * c_prev.b

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

    def add_phong_shading(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
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

        L = np.array(view_vector)

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):

                vec_k = np.arange(-100, 100, 1)
                x_k = vec_k * view_vector[0]
                y_k = vec_k * view_vector[1]
                z_k = vec_k * view_vector[2]

                vc_base_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + volume_center[0]
                vc_base_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + volume_center[1]
                vc_base_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + volume_center[2]

                vc_vec_x = vc_base_x + x_k
                vc_vec_y = vc_base_y + y_k
                vc_vec_z = vc_base_z + z_k

                vx = 0
                N = np.zeros(3)
                found = False
                for k in range(len(vec_k)):
                    vx = get_voxel(volume, vc_vec_x[k], vc_vec_y[k], vc_vec_z[k])
                    if vx > 1:
                        found = True
                        delta_f = self.annotation_gradient_volume.get_gradient(vc_vec_x[k], vc_vec_y[k], vc_vec_z[k])
                        magnitude_f = delta_f.magnitude
                        N[0] = delta_f.x / magnitude_f
                        N[1] = delta_f.y / magnitude_f
                        N[2] = delta_f.z / magnitude_f
                        break

                # Normalize value to be between 0 and 1
                if found:
                    shadow = (np.dot(N.reshape(1, 3), L) + 1) / 2
                else:
                    shadow = 0
                #shadow = math.floor(shadow * 255) if shadow < 255 else 255
                # Assign color to the pixel i, j
                if shadow < 220:
                    image[(j * image_size + i) * 4] *= shadow
                    image[(j * image_size + i) * 4 + 1] *= shadow
                    image[(j * image_size + i) * 4 + 2] *= shadow
                    image[(j * image_size + i) * 4 + 3] = 255

    def render_flat_surface(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()
        u_vector = view_matrix[0:3]
        v_vector = view_matrix[4:7]
        view_vector = view_matrix[8:11]
        image_center = image_size / 2
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]

        L = np.array(view_vector)

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):

                vec_k = np.arange(-100, 100, 1)
                x_k = vec_k * view_vector[0]
                y_k = vec_k * view_vector[1]
                z_k = vec_k * view_vector[2]

                vc_base_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + volume_center[0]
                vc_base_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + volume_center[1]
                vc_base_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + volume_center[2]

                vc_vec_x = vc_base_x + x_k
                vc_vec_y = vc_base_y + y_k
                vc_vec_z = vc_base_z + z_k

                vx = 0
                N = np.zeros(3)
                found = False
                for k in range(len(vec_k)):
                    vx = get_voxel(volume, vc_vec_x[k], vc_vec_y[k], vc_vec_z[k])
                    if vx > 1:
                        found = True
                        delta_f = self.annotation_gradient_volume.get_gradient(vc_vec_x[k], vc_vec_y[k], vc_vec_z[k])
                        magnitude_f = delta_f.magnitude
                        N[0] = delta_f.x / magnitude_f
                        N[1] = delta_f.y / magnitude_f
                        N[2] = delta_f.z / magnitude_f
                        break

                # Normalize value to be between 0 and 1
                if found:
                    shadow = (np.dot(N.reshape(1, 3), L) + 1) / 2
                    alpha = 255
                else:
                    shadow = 0
                    alpha = 0

                red = math.floor(shadow * 255) if shadow < 1 else 255
                green = math.floor(shadow * 255) if shadow < 1 else 255
                blue = math.floor(shadow * 255) if shadow < 1 else 255

                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    def colored_q_slice(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        self.clear_image()
        u_vector = view_matrix[0:3]
        v_vector = view_matrix[4:7]
        view_vector = view_matrix[8:11]
        image_center = image_size / 2
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):

                vec_k = np.arange(0, 2, 1)
                x_k = vec_k * view_vector[0]
                y_k = vec_k * view_vector[1]
                z_k = vec_k * view_vector[2]

                vc_base_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + volume_center[0]
                vc_base_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + volume_center[1]
                vc_base_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + volume_center[2]

                vc_vec_x = vc_base_x + x_k
                vc_vec_y = vc_base_y + y_k
                vc_vec_z = vc_base_z + z_k

                vx = 0
                N = np.zeros(3)
                for k in range(len(vec_k)):
                    vx = get_voxel(volume, vc_vec_x[k], vc_vec_y[k], vc_vec_z[k])
                    if vx > 1:
                        delta_f = self.annotation_gradient_volume.get_gradient(vc_vec_x[k], vc_vec_y[k], vc_vec_z[k])
                        magnitude_f = delta_f.magnitude
                        N[0] = delta_f.x / magnitude_f
                        N[1] = delta_f.y / magnitude_f
                        N[2] = delta_f.z / magnitude_f
                        break

                # Get colours for voxels
                voxel_colour = colour_table.loc[vx]

                if vx == 0:
                    red = 0
                    green = 0
                    blue = 0
                    alpha = 0
                else:
                    red = math.floor(voxel_colour.r * 255) if voxel_colour.r < 1 else 255
                    green = math.floor(voxel_colour.g * 255) if voxel_colour.g < 1 else 255
                    blue = math.floor(voxel_colour.b * 255) if voxel_colour.b < 1 else 255
                    alpha = 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha


    def render_mouse_brain(self, view_matrix: np.ndarray, annotation_volume: Volume, energy_volumes: dict,
                           image_size: int, image: np.ndarray, quick: bool = False):

        if quick:
            self.colored_q_slice(view_matrix, annotation_volume, image_size, image)
        else:
            self.render_annotation_compositing(view_matrix, annotation_volume, image_size, image)
            self.add_phong_shading(view_matrix, annotation_volume, image_size, image)
            # self.render_flat_surface(view_matrix, annotation_volume, image_size, image)



class GradientVolumeImpl(GradientVolume):
    def compute(self):
        # Implemented in volume.py
        pass
