import numpy as np
import math
from typing import List

class Volume:
    """
    Volume data class.

    Attributes:
        data: Numpy array with the voxel data. Its shape will be (dim_x, dim_y, dim_z).
        dim_x: Size of dimension X.
        dim_y: Size of dimension Y.
        dim_z: Size of dimension Z.
    """

    def __init__(self, array, compute_histogram=True):
        """
        Inits the volume data.
        :param array: Numpy array with shape (dim_x, dim_y, dim_z).
        """

        self.data = array
        self.histogram = np.array([])
        self.dim_x = array.shape[0]
        self.dim_y = array.shape[1]
        self.dim_z = array.shape[2]

        if compute_histogram:
            self.compute_histogram()

    def get_voxel(self, x, y, z):
        """Retrieves the voxel for the """
        return self.data[x, y, z]

    def get_minimum(self):
        return self.data.min()

    def get_maximum(self):
        return self.data.max()

    def compute_histogram(self):
        self.histogram = np.histogram(self.data, bins=np.arange(self.get_maximum() + 1))[0]


class VoxelGradient:
    def __init__(self, gx=0, gy=0, gz=0):
        self.x = gx
        self.y = gy
        self.z = gz
        self.magnitude = math.sqrt(gx * gx + gy * gy + gz * gz)


ZERO_GRADIENT = VoxelGradient()


class GradientVolume:
    def __init__(self, volume):
        self.volume = volume
        self.data : List[VoxelGradient] = []
        self.compute()
        self.max_magnitude = -1.0
        self.magnitude_volume = None

    def get_gradient(self, x, y, z):
        x = int(math.floor(x))
        y = int(math.floor(y))
        z = int(math.floor(z))
        return self.data[x + self.volume.dim_x * (y + self.volume.dim_y * z)]

    def set_gradient(self, x, y, z, value):
        self.data[x + self.volume.dim_x * (y + self.volume.dim_y * z)] = value

    def get_voxel(self, i):
        return self.data[i]

    def compute(self):
        """
        Computes the gradient for the current volume
        """
        # this just initializes all gradients to the vector (0,0,0)
        self.data = [ZERO_GRADIENT] * (self.volume.dim_x * self.volume.dim_y * self.volume.dim_z)

        for i in range(1, self.volume.dim_x-1):
            for j in range(1, self.volume.dim_y-1):
                for k in range(1, self.volume.dim_z-1):
                    d_x = 0.5 * (self.volume.get_voxel(i+1, j, k) - self.volume.get_voxel(i-1, j, k))
                    d_y = 0.5 * (self.volume.get_voxel(i, j+1, k) - self.volume.get_voxel(i, j-1, k))
                    d_z = 0.5 * (self.volume.get_voxel(i, j, k+1) - self.volume.get_voxel(i, j, k-1))
                    self.set_gradient(i, j, k, VoxelGradient(d_x, d_y, d_z))

    def get_magnitude_volume(self):
        if self.magnitude_volume is None:
            self.magnitude_volume = np.zeros((self.volume.dim_x, self.volume.dim_y, self.volume.dim_z))
            for i in range(1, self.volume.dim_x - 1):
                for j in range(1, self.volume.dim_y - 1):
                    for k in range(1, self.volume.dim_z - 1):
                        self.magnitude_volume[i][j][k] = self.get_gradient(i, j, k).magnitude
        return Volume(self.magnitude_volume)

    def get_max_gradient_magnitude(self):
        if self.max_magnitude < 0:
            gradient = max(self.data, key=lambda x: x.magnitude)
            self.max_magnitude = gradient.magnitude

        return self.max_magnitude