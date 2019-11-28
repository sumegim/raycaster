import numpy as np

from implementation import single_trilinear_interpolation


def test_simple_interpolation():
    # Vertices such that the front face of the cube is 1, the back face 0
    view_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    inverse_view_matrix = np.linalg.inv(view_matrix)
    vertices = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [
                        0, 0, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0]])

    # Point in the middle of the cube
    assert single_trilinear_interpolation(
        np.array([0.5, 0.5, 0.5]), vertices, inverse_view_matrix) == 0.5
    # Point in the middle of the front face
    assert single_trilinear_interpolation(
        np.array([0.5, 0.5, 0]), vertices, inverse_view_matrix) == 1
    # Point in the middle of the back face
    assert single_trilinear_interpolation(
        np.array([0.5, 0.5, 1]), vertices, inverse_view_matrix) == 0

    # Should work with messed up coords
    vertices = np.array([[1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 1], [
                        0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0]])

    # Point in the middle of the cube
    assert single_trilinear_interpolation(
        np.array([0.5, 0.5, 0.5]), vertices, inverse_view_matrix) == 0.5
    # Point in the middle of the front face
    assert single_trilinear_interpolation(
        np.array([0.5, 0.5, 0]), vertices, inverse_view_matrix) == 1
    # Point in the middle of the back face
    assert single_trilinear_interpolation(
        np.array([0.5, 0.5, 1]), vertices, inverse_view_matrix) == 0


def test_rotated_coords():
    view_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    inverse_view_matrix = np.linalg.inv(view_matrix)
    vertices = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [
                        0, 0, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0]])
    point = np.array([0.5, 0.5, 0.5])

    # Let's rotate everything
    rotated_view = np.array(
        [[0.10, -1, -0.12], [-0.8, -0.15, 0.53], [-0.5, -0.5, -0.83]])
    inverse_rotated_matrix = np.linalg.inv(rotated_view)
    rotated_vertices = np.concatenate(
        [(rotated_view @ vertices[:, :-1].T).T, vertices[:, -1:]], axis=1)
    rotated_point = rotated_view @ point

    assert single_trilinear_interpolation(
        point, vertices, inverse_view_matrix) == single_trilinear_interpolation(rotated_point, rotated_vertices, inverse_rotated_matrix), "interpolation should be rotation-invariant"
