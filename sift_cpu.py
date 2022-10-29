import numpy
from PIL import Image  # change for JS
import cv2
import numpy as np
import math
from functools import cmp_to_key


float_tolerance = 1e-7


def visualize_images(pyramid):
    cnt = 0
    for i in pyramid:
        for j in i:
            j = cv2.cvtColor(j, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'temp{cnt}.jpeg', j)
            cnt += 1


def visualize_keypoints(image, keypoints):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for kp in keypoints:
        i, j = kp.pt
        print(i, j)
        image = cv2.circle(image, (int(i), int(j)), radius=2, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(f'temp_kpr.jpeg', image)


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def generate_base(image, sigma=1.6, assumed_blur=.5):
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)  # TODO: parallizable check rezie
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    print(sigma_diff)
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff,
                            sigmaY=sigma_diff)  # TODO: Prarallelizable Gausian Function


def compute_levels(image_shape):
    return int(math.log(min(image_shape), 2) - 1)  # TODO: REVIEW -1


'''Check paper for kernel level iml https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf'''


def generate_kernels(sigma, intervals):
    num_image_per_level = intervals + 3
    k = 2 ** (1.0 / intervals)
    gaussian_kernels = np.zeros(num_image_per_level)
    gaussian_kernels[0] = sigma
    sigma_previous = sigma

    for index in range(1, num_image_per_level):
        sigma_total = k * sigma_previous
        gaussian_kernels[index] = math.sqrt(sigma_total ** 2 - sigma_previous ** 2)
        sigma_previous = sigma_total
    return gaussian_kernels


def generate_gaussians(image, levels, kernels):  # TODO: Easily Parallelizable once Gaussian Blur is parallelized
    gaussian_images = []

    for level in range(levels):
        gaussian_image_level = [image]  # first image is blurred correctly
        for kernel in kernels[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=kernel, sigmaY=kernel)
            gaussian_image_level.append(image)

        gaussian_images.append(np.array(gaussian_image_level))
        level_base = gaussian_image_level[-3]
        image = cv2.resize(level_base, (level_base.shape[1] // 2, level_base.shape[0] // 2),
                           interpolation=cv2.INTER_NEAREST)
    return gaussian_images


def generate_difference_gaussian(gausian_images):  # TODO: Easily Parallelizable subtraction
    difference_gaussian = []
    for level in gausian_images:
        DoG_level = []
        for i in range(len(level) - 1):
            DoG = cv2.subtract(level[i + 1], level[i])
            DoG_level.append(DoG)
        difference_gaussian.append(DoG_level)

    visualize_images(difference_gaussian)
    return difference_gaussian


def compute_scale_extreme(gaussian_images, difference_gaussian, intervals, sigma, image_border_width,
                          contrast_threshold=0.04):
    threshold = math.floor(0.5 * contrast_threshold / intervals * 255)
    key_points = []

    for level, DoG_images in enumerate(difference_gaussian):
        for index, (im1, im2, im3) in enumerate(zip(DoG_images, DoG_images[1:], DoG_images[2:])):
            # (i,j) is the center of the cube
            for i in range(image_border_width, im1.shape[0] - image_border_width):
                # TODO: independent look can be parralelized
                for j in range(image_border_width, im1.shape[1] - image_border_width):
                    if is_pixel_extreme(im1[i - 1:i + 2, j - 1:j + 2], im2[i - 1:i + 2, j - 1:j + 2],
                                        im3[i - 1:i + 2, j - 1:j + 2], threshold):
                        localization_result = localize_extreme_quadratic_fit(i, j, index + 1, level,
                                                                             intervals, DoG_images,
                                                                             sigma, contrast_threshold,
                                                                             image_border_width)

                        if localization_result is not None:
                            keypoint, localization_index = localization_result
                            key_points_with_orientation = compute_keypoints_orientations(keypoint, level,
                                                                                         gaussian_images[level][
                                                                                             localization_index])
                            for keypoint_with_orientation in key_points_with_orientation:
                                key_points.append(keypoint_with_orientation)
    return key_points


def is_pixel_extreme(im1, im2, im3, threshold=0.04):
    """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise
    """
    center_pixel = im2[1, 1]
    if abs(center_pixel) > threshold:
        if center_pixel > 0:
            return np.max(np.array([im1, im2, im3])) == center_pixel
        elif center_pixel < 0:
            return np.min(np.array([im1, im2, im3])) == center_pixel
    return False


def localize_extreme_quadratic_fit(i, j, index, level, intervals, dog_images, sigma, contrast_threshold,
                                   image_border_width, eigen_ratio=10, convergence_attempt=5):
    # init
    pixel_cube = np.array([0, 0, 0])
    hessian = np.array([0, 0, 0])
    gradient = np.array([0, 0, 0])
    extreme_update = np.array([0, 0, 0])
    extreme_is_outside = 0
    image_shape = dog_images[0].shape
    try_index = 0

    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
    """
    for try_index in range(convergence_attempt):
        im1, im2, im3 = dog_images[index - 1:index + 2]
        pixel_cube = np.stack([im1[i - 1:i + 2, j - 1:j + 2],
                               im2[i - 1:i + 2, j - 1:j + 2],
                               im3[i - 1:i + 2, j - 1:j + 2]]).astype('float32') / 255.
        gradient = compute_gradient_center(pixel_cube)
        hessian = compute_hessian_center(pixel_cube)
        extreme_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if abs(extreme_update[0]) < 0.5 and abs(extreme_update[1]) < 0.5 and abs(extreme_update[2]) < 0.5:
            break

        j += int(round(extreme_update[0]))
        i += int(round(extreme_update[1]))
        index += int(round(extreme_update[2]))

        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= \
                image_shape[1] - image_border_width or index < 1 or index > intervals:
            extreme_is_outside = True
            break

    if extreme_is_outside:
        # Updated extremum moved outside of image before reaching convergence. Skipping...
        return None

    if try_index >= convergence_attempt - 1:
        # Exceeded maximum number of attempts without reaching convergence for this extremum. Skipping...

        return None

    value_after_update = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extreme_update)
    if np.abs(value_after_update) * intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)
        if xy_hessian_det > 0 and eigen_ratio * (xy_hessian_trace ** 2) < ((eigen_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + extreme_update[0]) * (2 ** level), (i + extreme_update[1]) * (2 ** level))
            keypoint.octave = level + index * (2 ** 8) + int(round((extreme_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((index + extreme_update[2]) / np.float32(intervals))) * (2 ** (level + 1))
            keypoint.response = np.abs(value_after_update)
            return keypoint, index


def compute_gradient_center(pixel_cube):
    """check paper for finite approximation https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
    Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order
    O(D^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(D^2) for f'(x) is (f(x + D) - f(x - D)) / (2 * D)
    # Here D = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    dx = 0.5 * (pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0])
    dy = 0.5 * (pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1])
    ds = 0.5 * (pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1])
    return np.array([dx, dy, ds])


def compute_hessian_center(pixel_cube):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(delta^2)
    With step size delta, the central difference formula of order O(D^2) for
            f''(x) is (f(x + D) - 2 * f(x) + f(x - D)) / (D ^ 2)
    Here D = 1, so the formula simplifies to
            f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    With step size D, the central difference formula of order
            O(D^2) for (d^2) f(x, y) / (dx dy) = (f(x + D y + D) - f(x + D, y - D) - f(x - D, y + D) + f(x - D, y - D)) / (4 * h ^ 2)
    Here D = 1, so the formula simplifies to
            (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    x corresponds to second array axis,
    y corresponds to first array axis,
    s (scale) corresponds to third array axis"""
    center_pixel = pixel_cube[1, 1, 1]
    dxx = pixel_cube[1, 1, 2] - 2 * center_pixel + pixel_cube[1, 1, 0]
    dyy = pixel_cube[1, 2, 1] - 2 * center_pixel + pixel_cube[1, 0, 1]
    dss = pixel_cube[2, 1, 1] - 2 * center_pixel + pixel_cube[0, 1, 1]
    dxy = 0.25 * (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0])
    dxs = 0.25 * (pixel_cube[2, 1, 2] - pixel_cube[2, 1, 0] - pixel_cube[0, 1, 2] + pixel_cube[0, 1, 0])
    dys = 0.25 * (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1])
    return np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])


def compute_keypoints_orientations(keypoint, level, image, radius_factor=3, num_bins=36, peak_ratio=0.8,
                                   scale_factor=1.5):
    keypoint_with_orientation = []
    image_shape = image.shape

    scale = scale_factor * keypoint.size / np.float32(2 ** (level + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)
    for i in range(-radius, radius + 1):
        # create a histogram of gradients for pixels around the keypoint’s neighborhood.
        # The neighborhood will cover pixels within 3 * scale of each keypoint
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** level))) + i
        if 0 < region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** level))) + j
                if 0 < region_x < image_shape[1] - 1:
                    dx = image[region_y, region_x + 1] - image[region_y, region_x - 1]
                    dy = image[region_y - 1, region_x] - image[region_y + 1, region_x]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    # create a 36-bin histogram for the orientations — 10 degrees per bin.
                    # The orientation of a particular pixel tells which histogram bin to choose,
                    # but the actual value placed in that bin is that pixel’s gradient magnitude
                    # with a Gaussian weighting.This makes pixels farther from the keypoint have
                    # less of an influence on the histogram
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        # smooth the histogram. The smoothing coefficients correspond to a 5-point Gaussian filtersmooth the histogram.
        # The smoothing coefficients correspond to a 5-point Gaussian filter.
        # https://theailearner.com/2019/05/06/gaussian-blurring/
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) +
                               raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.

    orientation_max = np.max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1),
                                                smooth_histogram > np.roll(smooth_histogram, -1)))[0]

    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (
                    left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoint_with_orientation.append(new_keypoint)
    return keypoint_with_orientation


def compare_keypoints(keypoint1, keypoint2):
    """Return keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id


def convery_keypoint_to_img_size(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints


def remove_duplicate(keypoints):
    """Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compare_keypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique = unique_keypoints[-1]
        if last_unique.pt[0] != next_keypoint.pt[0] or last_unique.pt[1] != next_keypoint.pt[1] or \
                last_unique.size != next_keypoint.size or last_unique.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints


def unpack_level(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    level = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if level >= 128:
        level = level | -128
    scale = 1 / np.float32(1 << level) if level >= 0 else np.float32(1 << -level)
    return level, layer, scale


def generate_descriptors(key_points, pyramid, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint
    """
    descriptors = []
    for keypoint in key_points:
        octave, layer, scale = unpack_level(keypoint)

        gaussian_image = pyramid[octave + 1][layer]
        num_rows, num_cols = gaussian_image.shape
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        # first two dimensions are increased by 2 to account for border effects
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list,
                                                                orientation_bin_list):
            # Smoothing via trilinear interpolation
            # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
            # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(
                int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), float_tolerance)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')


def SIFT(image, sigma=1.6, assumed_blur=.5, num_intervals=3, image_border_width=5, visualize=0):
    image = image.astype('float32')
    image_base = generate_base(image, sigma, assumed_blur)
    levels = compute_levels(image_base.shape[:2])
    kernels = generate_kernels(sigma, num_intervals)
    print(levels)
    print(kernels)
    gaussian_images = generate_gaussians(image_base, levels, kernels)
    difference_gaussian = generate_difference_gaussian(gaussian_images)
    key_points = compute_scale_extreme(gaussian_images, difference_gaussian, num_intervals, sigma, image_border_width)
    print(key_points)
    key_points = remove_duplicate(key_points)
    key_points = convery_keypoint_to_img_size(key_points)
    if np.any(visualize):
        visualize_keypoints(visualize, key_points)
    descriptors = generate_descriptors(key_points, gaussian_images)
    return key_points,descriptors


if __name__ == '__main__':
    img = Image.open('test.jpg')
    img = numpy.asarray(img)
    img_gray = rgb2gray(img)
    a = SIFT(img_gray, visualize=img)
    print(a)
    # a = cv2.cvtColor(a,cv2.COLOR_RGB2BGR)
    # cv2.imwrite('temp.jpeg', a)
