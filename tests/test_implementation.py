import numpy as np

from implementation import single_trilinear_interpolation

def test_single_trilinear_interpolation():
    # Vertices such that the front face of the cube is 1, the back face 0
    vertices = np.array([[0,0,0,1],[1,0,0,1],[0,1,0,1],[1,1,0,1],[0,0,1,0],[1,0,1,0],[0,1,1,0],[1,1,1,0]])

    # Point in the middle of the cube
    assert single_trilinear_interpolation(np.array([0.5,0.5,0.5]), vertices) == 0.5
    # Point in the middle of the front face
    assert single_trilinear_interpolation(np.array([0.5,0.5,0]), vertices) == 1
    # Point in the middle of the back face
    assert single_trilinear_interpolation(np.array([0.5,0.5,1]), vertices) == 0

    # Should work with messed up coords
    vertices = np.array([[1,0,0,1],[0,0,1,0],[0,1,0,1],[1,1,0,1],[0,1,1,0],[1,0,1,0],[0,0,0,1],[1,1,1,0]])

    # Point in the middle of the cube
    assert single_trilinear_interpolation(np.array([0.5,0.5,0.5]), vertices) == 0.5
    # Point in the middle of the front face
    assert single_trilinear_interpolation(np.array([0.5,0.5,0]), vertices) == 1
    # Point in the middle of the back face
    assert single_trilinear_interpolation(np.array([0.5,0.5,1]), vertices) == 0
