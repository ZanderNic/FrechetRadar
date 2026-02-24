# std lib imports


# 3 party import
import numpy as np

# projekt imports
from RadarDataGen.Data_Generator.pseudo_radar_points import pseudo_radar_points
from RadarDataGen.Discretizer.radar_discretizer import RadarDiscretizer

# Test Params ########################################

discretizer = RadarDiscretizer(32, -1, 1, -1, 1, 1.0)
axis_limits = [-1, 1, -1, 1]

lambda_lines_2d=5
lambda_points_per_line_2d=20
lambda_clutter=10

######################################################


if __name__ == "__main__":
    points = pseudo_radar_points(lambda_lines_2d=lambda_lines_2d, lambda_points_line_2d=lambda_points_per_line_2d, lambda_clutter=lambda_clutter)
    grid = discretizer.points_to_grid(points)
    points_from_grid = discretizer.grid_to_points(grid)

    error = np.linalg.norm(points[np.lexsort(points.T)] - points_from_grid[np.lexsort(points_from_grid.T)])
    print("The Reconstruction error from points to grid and then back from grid to points should be 0")
    print(f'Reconstruction error: {error:.6f}')
