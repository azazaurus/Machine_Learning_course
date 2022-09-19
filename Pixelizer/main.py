from matplotlib import pyplot as plt
from matplotlib.image import imread, imsave
import numpy as np


def get_average_vector(matrix_section):
    matrix_section = matrix_section.reshape(-1, 3)
    average_matrix_columns = np.mean(matrix_section, axis=0)
    int_average_matrix_columns = np.round(average_matrix_columns).astype(int)
    return int_average_matrix_columns


def get_coordinates(min_, max_, segments_count):
    area_length = max_ - min_ + 1
    enlarged_segments_count = area_length % segments_count
    segment_size = area_length // segments_count
    coordinates = [min_]

    for i in range(segments_count):
        coordinates.append(coordinates[i] + segment_size)

        if i == segments_count - enlarged_segments_count - 1:
            segment_size = segment_size + 1

    return coordinates


def change_submatrix_value(matrix, vector, y_min, y_max, x_min, x_max):
    changed_matrix = np.copy(matrix)
    vector_length = len(vector)
    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            for k in range(vector_length):
                changed_matrix[i, j, k] = vector[k]

    return changed_matrix


def pixelize(image, y_min, y_max, x_min, x_max, h):
    orig_img = imread(image)
    plt.imshow(orig_img)
    plt.show()
    pixelized_matrix = np.copy(orig_img)

    y_coordinates = get_coordinates(y_min, y_max, h)
    x_coordinates = get_coordinates(x_min, x_max, h)

    for i, y in enumerate(y_coordinates[:-1]):
        for j, x in enumerate(x_coordinates[:-1]):
            pixel = orig_img[y_coordinates[i]:y_coordinates[i+1], x_coordinates[j]:x_coordinates[j+1]]
            average_vector = get_average_vector(pixel)
            pixelized_matrix = change_submatrix_value(pixelized_matrix, average_vector, y_coordinates[i], y_coordinates[i + 1], x_coordinates[j], x_coordinates[j + 1])

    plt.imshow(pixelized_matrix)
    plt.show()
    return pixelized_matrix


if __name__ == '__main__':
    pixelized_matrix = pixelize('Face-Mask-Woman.jpeg', 2200, 2600, 3300, 4900, 10)
    imsave('pixelized_image.jpeg', pixelized_matrix)

