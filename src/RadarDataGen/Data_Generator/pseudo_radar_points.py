# std lib imports
import time
from typing import Optional

# 3 party import
import numpy as np


########################################################################

def pseudo_radar_lines_2d(
        lambda_lines: int,
        lambda_points_line : int,
        rng : np.random.Generator,
):
    """
        This is a very basic pseudo radar data generator as presented by Jonas Neuhöfer and Jörg Reichardt. It creates points in x, y with a random uniform third 
        dim.

        Points a located on a lines. The number of lines is drawn from a Poisson distribution
        with mean lambda_lines. Orientiation of lines is random. For each line, a number of points
        is drawn from a Poisson distribution with mean lambda_points_per_line. 
        Additionally, a the number of clutter points is drawn from a Poisson distribution with
        mean lambda_clutter. Clutter points are uniformly distributed.
        Points are returned in random order.
    """
    num_lines = rng.poisson(lambda_lines)
    num_points_per_line = rng.poisson(lambda_points_line, size = num_lines)
    total_points = np.sum(num_points_per_line)
    if total_points == 0:
        return np.empty((0, 3), dtype=np.float32)

    m = rng.uniform(-1, 1, num_lines)         # sample m and the t of all lines
    t = rng.uniform(-1, 1, num_lines)
    x = rng.uniform(-1, 1, total_points)      # smaple the x for all points of all lines
    color = rng.random(num_lines)
    
    line_idx = np.repeat(np.arange(num_lines), num_points_per_line)
    x = rng.uniform(-1, 1, total_points)
    y = t[line_idx] + m[line_idx] * x
    c = color[line_idx]

    points = np.column_stack((x, y, c)).astype(np.float32)

    return points



# def pseudo_radar_lines_2d(lambda_lines: int, lambda_points_line: int):
#     """
#         Vectorized version of pseudo radar line generator.

#         Each line is defined by two random endpoints inside [-1,1]^2.
#         Points are sampled along the segment so that all resulting points
#         stay inside [-1,1]^2*.

#         Returns:
#             (N, 3) array: x, y, color
#     """
#     num_lines = np.random.poisson(lambda_lines)
#     if num_lines == 0:
#         return np.empty((0, 3), dtype=np.float32)

#     n_per_line = np.random.poisson(lambda_points_line, size=num_lines)
#     total_points = np.sum(n_per_line)
#     if total_points == 0:
#         return np.empty((0, 3), dtype=np.float32)

#     colors = np.random.rand(num_lines, 1)

#     P1 = np.random.uniform(-1, 1, size=(num_lines, 2))
#     P2 = np.random.uniform(-1, 1, size=(num_lines, 2))

#     x1 = P1[:, 0]
#     y1 = P1[:, 1]
#     x2 = P2[:, 0]
#     y2 = P2[:, 1]

#     vertical = np.isclose(x1, x2)
#     non_vertical = ~vertical
#     x_min = np.maximum(-1, np.minimum(x1, x2))
#     x_max = np.minimum( 1, np.maximum(x1, x2))
#     line_index = np.repeat(np.arange(num_lines), n_per_line)
#     x = np.random.uniform(x_min[line_index],
#                           x_max[line_index])
#     y = np.empty_like(x)
#     idx_v = vertical[line_index]
#     y[idx_v] = np.random.uniform(
#         np.minimum(y1[line_index[idx_v]], y2[line_index[idx_v]]),
#         np.maximum(y1[line_index[idx_v]], y2[line_index[idx_v]])
#     )
#     x[idx_v] = x1[line_index[idx_v]]
#     idx_nv = ~idx_v
#     m = (y2 - y1) / (x2 - x1 + 1e-12)
#     t = y1 - m * x1
#     y[idx_nv] = m[line_index[idx_nv]] * x[idx_nv] + t[line_index[idx_nv]]
#     y = np.clip(y, -1, 1)
#     c = colors[line_index].flatten()
#     points = np.column_stack((x, y, c)).astype(np.float32)
#     return points


def pseudo_radar_lines_3d(lambda_lines: int, lambda_points_line: int, rng: np.random.Generator):
    """
        Generates 3D radar points lying on random lines in R3.
        Each line is defined by a random point and a random direction.
    """
    num_lines = rng.poisson(lambda_lines)
    num_points_per_line = rng.poisson(lambda_points_line, size=num_lines)

    total_points = np.sum(num_points_per_line)
    directions = rng.uniform(-1, 1, (num_lines, 3))   # direction
    origins = rng.uniform(-1, 1, (num_lines, 3))      # vector to our start point
    scalars = rng.uniform(-1, 1, total_points)        # scalers because a parametric line in R3 is def by  y = vector_on_line + x * direction_vector

    points = np.empty((total_points, 3), dtype=np.float32)
    idx = 0
    for i in range(num_lines):
        for _ in range(num_points_per_line[i]):
            points[idx] = origins[i] + scalars[idx] * directions[i]
            idx += 1

    return points


def pseudo_radar_clutter(
    lambda_clutter: int,
    rng : np.random.Generator
):
    """
        This function samples radar clutter in 3 dim uniformly
    """
    num_clutter_points = rng.poisson(lambda_clutter)
    points = rng.uniform(-1, 1, (num_clutter_points, 2)) # sample the clutter 
    colors = rng.random((num_clutter_points, 1))
    points = np.hstack([points, colors])

    return points


def pseudo_radar_rectangle_3d(lambda_rectangle: int, lambda_points_rectangle: int, rng : np.random.Generator):
    """
        Samples random rectangles (2D surfaces) in R3. Each rectangle is defined by a point and two direction vectors.
    """
    points = np.empty([0, 3])
    
    num_rectangle = rng.poisson(lambda_rectangle)
    num_points_per_rectangle =rng.poisson(lambda_points_rectangle, size=num_rectangle)

    for i in range(num_rectangle):
        origin = rng.uniform(-0.5, 0.5, 3)
        v1 = rng.uniform(-0.5, 0.5, 3)
        v2 = rng.uniform(-0.5, 0.5, 3)

        for _ in range(num_points_per_rectangle[i]):
            alpha, beta = rng.uniform(0, 1, 2)
            y = origin + alpha * v1 + beta * v2
            points = np.vstack([points, y], dtype=np.float32)

    return points


def pseudo_radar_rectangle_2d(lambda_rectangle: int, lambda_points_rectangle: int, rng : np.random.Generator):
    """
        Samples random rectangles (2D surfaces) in R2. Each rectangle is defined by a starting point (x, y),
        a horizontal length, a vertical height, and a color value. Points are sampled uniformly within each rectangle.
        
        Parameters:
        - lambda_rectangle: Expected number of rectangles (Poisson-distributed).
        - lambda_points_rectangle: Expected number of points per rectangle (Poisson-distributed).
        
        Returns:
        - points: A NumPy array of shape (N, 3), where each row is (x, y, color).
    """
    num_rectangle = rng.poisson(lambda_rectangle)
    num_points_per_rectangle = rng.poisson(lambda_points_rectangle, size=num_rectangle)
    
    total_points = sum(num_points_per_rectangle)
    points = np.empty([total_points, 3], dtype=np.float32)

    start_points = rng.uniform(-1, 0.5, num_rectangle * 2)
    angles = rng.uniform(0, 2 * np.pi, num_rectangle)
    x_len = rng.uniform(0.1, 1, num_rectangle)
    y_len = rng.uniform(0.1, 1, num_rectangle)
    color = rng.random(num_rectangle)

    idx = 0
    for i in range(num_rectangle):
        # Rotation matrix for the rectangle
        R = np.array([[np.cos(angles[i]), -np.sin(angles[i])],
                      [np.sin(angles[i]),  np.cos(angles[i])]])

        lambda_x = rng.uniform(0, 2, num_points_per_rectangle[i])
        lambda_y = rng.uniform(0, 2, num_points_per_rectangle[i])

        for j in range(num_points_per_rectangle[i]):
            x = start_points[2*i] +  lambda_x[j] * x_len[i]
            y = start_points[2*i + 1] +  lambda_y[j] * y_len[i]
            
            global_point = R @ np.array([x, y])
            points[idx] = (global_point[0], global_point[1], color[i])
            idx += 1

    return points


def pseudo_radar_rect_outline_2d(lambda_rect_outline: int, lambda_points_rect_outline: int, rng : np.random.Generator):
    """
        Samples points on the outline of randomly oriented rectangles in 2D.
        Each rectangle is defined by a center point, width, height, and rotation angle.

        Parameters:
        - lambda_rect_outline: Expected number of rectangles (Poisson-distributed).
        - lambda_points_rect_outline: Expected number of points per rectangle (Poisson-distributed).

        Returns:
        - points: A NumPy array of shape (N, 3), where each row is (x, y, color).
    """
    num_rectangles = rng.poisson(lambda_rect_outline)
    num_points_per_rectangle =rng.poisson(lambda_points_rect_outline, size=num_rectangles)

    total_points = np.sum(num_points_per_rectangle)
    points = np.empty((total_points, 3), dtype=np.float32)

    # Rectangle properties
    centers = rng.uniform(-1, 1, (num_rectangles, 2))   # rectangle centers
    widths = rng.uniform(0.2, 1.0, num_rectangles)      # rectangle width
    heights = rng.uniform(0.2, 1.0, num_rectangles)     # rectangle height
    angles = rng.uniform(0, 2 * np.pi, num_rectangles)  # rotation angle
    colors = rng.random(num_rectangles)

    idx = 0
    for i in range(num_rectangles):
        cx, cy = centers[i]
        w, h = widths[i], heights[i]
        theta = angles[i]

        # Rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])

        for _ in range(num_points_per_rectangle[i]):
            edge = rng.integers(0, 4)  # choose one of the four edges
            t = rng.uniform(0, 1)     # position along the edge

            # Local coordinates before rotation
            if edge == 0:  # top edge
                local_point = np.array([-w/2 + t*w, h/2])
            elif edge == 1:  # bottom edge
                local_point = np.array([-w/2 + t*w, -h/2])
            elif edge == 2:  # left edge
                local_point = np.array([-w/2, -h/2 + t*h]) 
            else:  # right edge
                local_point = np.array([w/2, -h/2 + t*h])

            # Rotate and translate to global coordinates
            global_point = R @ local_point + np.array([cx, cy])
            points[idx] = (global_point[0], global_point[1], colors[i])
            idx += 1

    return points


def pseudo_radar_circles_2d(lambda_circles: int, lambda_points_circle: int, rng : np.random.Generator):
    """
        Samples random circles (2D surfaces) in R². Each circle is defined by a center point (x, y),
        a radius, and a color value. Points are sampled uniformly within each circle.
        
        Parameters:
        - lambda_circles: Expected number of circles (Poisson-distributed).
        - lambda_points_circle: Expected number of points per circle (Poisson-distributed).
        
        Returns:
        - points: A NumPy array of shape (N, 3), where each row is (x, y, color).
    """
    num_circles = rng.poisson(lambda_circles)
    num_points_per_circle = rng.poisson(lambda_points_circle, size=num_circles)
    
    total_points = sum(num_points_per_circle)
    points = np.empty([total_points, 3], dtype=np.float32)

    centers = rng.uniform(-1, 1, num_circles * 2)
    radius = rng.uniform(0.1, 0.5, num_circles)
    color = rng.random(num_circles)

    idx = 0
    for i in range(num_circles):
        # Sample points uniformly within a circle using polar coordinates
        r = radius[i] * np.sqrt(rng.uniform(0, 1, num_points_per_circle[i]))
        theta = rng.uniform(0, 2 * np.pi, num_points_per_circle[i])
        x = centers[2*i] + r * np.cos(theta)
        y = centers[2*i + 1] + r * np.sin(theta)

        for j in range(num_points_per_circle[i]):
            points[idx] = x[j], y[j], color[i]
            idx += 1

    return points

######## Here we define a combined generator as kombination of the above 

def pseudo_radar_points(
        lambda_lines_2d: int = 0,
        lambda_points_line_2d: int = 0,
        lambda_lines_3d: int = 0,
        lambda_points_line_3d: int = 0,
        lambda_rectangle_2d: int = 0, 
        lambda_points_rectangle_2d: int = 0,
        lambda_rect_outline_2d: int = 0,
        lambda_points_rect_outline_2d: int = 0,
        lambda_rectangle_3d: int = 0, 
        lambda_points_rectangle_3d: int = 0,
        lambda_circle: int = 0,
        lambda_points_circle: int = 0,
        lambda_clutter: int = 0,
        seed: int = None,
):
    """
        Generates synthetic 3D radar points based on a combination of geometric primitives:
        lines in 2D/3D, rectangles, and clutter.

        Parameters:
            lambda_lines_2d (int): Expected number of 2D lines.
            lambda_points_line_2d (int): Expected number of points per 2D line.
            lambda_lines_3d (int): Expected number of 3D lines.
            lambda_points_line_3d (int): Expected number of points per 3D line.
            lambda_rectangle (int): Expected number of rectangles.
            lambda_points_rectangle (int): Expected number of points per rectangle.
            lambda_clutter (int): Expected number of clutter points.

        Returns:
            np.ndarray: Array of shape (N, 3) containing all generated radar points.
    """
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)

    points = np.empty([0, 3], dtype=np.float32)

    if lambda_lines_2d and lambda_points_line_2d:
        points = np.vstack([points, pseudo_radar_lines_2d(lambda_lines_2d, lambda_points_line_2d, rng)])
    
    if  lambda_lines_3d and lambda_points_line_3d:
        points = np.vstack([points, pseudo_radar_lines_3d(lambda_lines_3d, lambda_points_line_3d, rng)])
    
    if lambda_rectangle_2d and lambda_points_rectangle_2d:
        points = np.vstack([points, pseudo_radar_rectangle_2d(lambda_rectangle_2d, lambda_points_rectangle_2d, rng)])

    if lambda_rect_outline_2d and lambda_points_rect_outline_2d:
        points = np.vstack([points, pseudo_radar_rect_outline_2d(lambda_rect_outline_2d, lambda_points_rect_outline_2d, rng)])

    if lambda_rectangle_3d and lambda_points_rectangle_3d:
        points = np.vstack([points, pseudo_radar_rectangle_3d(lambda_rectangle_3d, lambda_points_rectangle_3d, rng)])

    if lambda_circle and lambda_points_circle:
        points = np.vstack([points, pseudo_radar_circles_2d(lambda_circle, lambda_points_circle, rng)])

    if lambda_clutter:
        points = np.vstack([points, pseudo_radar_clutter(lambda_clutter, rng)])

    rng.shuffle(points)

    return points


def _pseudo_radar_points_with_info(lambda_lines: int, lambda_points_line: int, lambda_clutter: int = 0):
    """
    """
    num_lines = np.random.poisson(lambda_lines)
    num_points_per_line = np.random.poisson(lambda_points_line, size = num_lines)
    total_points = np.sum(num_points_per_line)
    if total_points == 0:
        return np.empty((0, 3), dtype=np.float32)

    m = np.random.uniform(-1, 1, num_lines)         # sample m and the t of all lines
    t = np.random.uniform(-1, 1, num_lines)
    x = np.random.uniform(-1, 1, total_points)      # smaple the x for all points of all lines
    color = np.random.random(num_lines)
    
    line_idx = np.repeat(np.arange(num_lines), num_points_per_line)
    x = np.random.uniform(-1, 1, total_points)
    y = t[line_idx] + m[line_idx] * x
    c = color[line_idx]

    points = np.column_stack((x, y, c)).astype(np.float32)


    # add clutter points 
    if lambda_clutter:
        clutter_points = pseudo_radar_clutter(lambda_clutter)
        points = np.vstack([points, clutter_points])

    return points, {"num_lines": num_lines, "num_points_per_line": num_points_per_line, "num_clutter": len(clutter_points)}