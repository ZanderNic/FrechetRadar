# std lib imports
from typing import List, Tuple, Dict
import time

# 3 party import
import numpy as np
from scipy.optimize import linear_sum_assignment

# projekt imports



class GridNeighbors:
    """
        This class stores neighbor cells for a 2D grid of size (width x height). For each cell, it caches neighbors for different 
        radiuses by saving the nummber of points that belong to each radius.
        
        Example:
            rad_index = [1, 5, 13] for a given cell means:
                - radius 0: neighbors of cell [:1]
                - radius 1: neighbors of cell [:5]
                - radius 2: neighbors of cell [:13]

        The class dosent stop caching when very big radiuses are provided but the worst case storage is O(width**2 * height **2)

        Parameter:
            width (int): 
                The width of the grid
            height (int):
                The height of the grid
    """

    def __init__(
        self, 
        width: int, 
        height: int,
        neighbor_method: str = "circle" # "circle" or "square" to prep the neighbor indexe 
    ):
        self.width = width
        self.height = height
        self.size = width * height

        # For each cell: neighbors list and cumulative rad_index
        self._neighbors: List[List[Tuple[int, int]]] = [[(x, y)] for y in range(height) for x in range(width)]
        self._dist_index: List[List[int]] = [[1] for _ in range(self.size)]

        if neighbor_method == "circle":
            self.prepared_radius = 0
            self.prepare_up_to = self.prepare_up_to_circle

        elif neighbor_method == "square":
            self.prepared_square_size = 0
            self.prepare_up_to = self.prepare_up_to_square

        else:
            raise ValueError(f"neighbor_method should be circle or square but is {neighbor_method}")
         

    def prepare_up_to_circle(
        self, 
        radius: int
    ):
        """
            This function prepares the neighbors list up to the given radius if the neighbors list is already prep for this radius return
        """
        if radius <= self.prepared_radius:
            return

        for rad in range(self.prepared_radius + 1, radius + 1):
            
            offsets = []

            for div_x in range(-rad, rad + 1):
                div_y = rad - abs(div_x)
                
                if div_y == 0:
                    offsets.append((div_x, 0))
                else:
                    offsets.append((div_x, div_y))
                    offsets.append((div_x, -div_y))
            
            for idx in range(self.size):
                x, y = self._index_to_coords(idx)
                new_neighbors = []
                for div_x, div_y in offsets:
                    new_x, new_y = int(x + div_x), int(y + div_y)
                    
                    if 0 <= new_x < self.width and 0 <= new_y < self.height:
                        new_neighbors.append((new_x, new_y))
                
                self._neighbors[idx].extend(new_neighbors)    
                self._dist_index[idx].append(len(self._neighbors[idx]))
            
            self.prepared_radius = rad
    

    def prepare_up_to_square(
        self, 
        square_size: int
    ):
        """
            This function prepares the neighbors list up to the given square_size if the neighbors list is already prep for this square_size return
        """
        if square_size <= self.prepared_square_size:
            return

        for size in range(self.prepared_square_size + 1, square_size + 1):
            
            rng = np.arange(-size, size + 1)
            dx, dy = np.meshgrid(rng, rng, indexing='xy')
            mask = (np.maximum(np.abs(dx), np.abs(dy)) == size)
            offsets = np.stack([dx[mask], dy[mask]], axis=1) 
            
            for idx in range(self.size):
                x, y = self._index_to_coords(idx)
                new_neighbors = []

                new_x = x + offsets[:, 0]
                new_y = y + offsets[:, 1]

                valid = (
                    (new_x >= 0) & (new_x < self.width) &
                    (new_y >= 0) & (new_y < self.height)
                )

                if np.any(valid):
                    new_neighbors = np.stack([new_x[valid], new_y[valid]], axis=1)  
              
                self._neighbors[idx].extend(map(tuple, new_neighbors))
                self._dist_index[idx].append(len(self._neighbors[idx]))

            self.prepared_square_size = size


    def get_neighbors(
        self, 
        cells: np.array, 
        dist: int
    ) -> np.array:
        
        if dist <= 0 or dist >= max(self.width, self.height):
            raise ValueError(f"radius is {dist} but needs to be > 0 and smaler than max(self.width, self.height) for our usecase")

        self.prepare_up_to(dist)
        result = set()

        for cell in cells:
            x, y = int(cell[0]), int(cell[1])
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"cell {(x, y)} is out of grid bounds")
            idx = self._coords_to_index(x, y)
            cutoff = self._dist_index[idx][dist]
            result.update(self._neighbors[idx][:cutoff])
        return np.array(list(result))


    def _coords_to_index(self, x: int, y: int) -> int:
        return y * self.width + x


    def _index_to_coords(self, idx: int) -> Tuple[int, int]:
        return (idx % self.width, idx // self.width)


class RadarDiscretizer:
    """
    
        Parameters:
            grid_size (int):
                how big the grid will be where all radar points will be discretised to
            x_min (float):
                the min value of x
            x_max (float):
                the max value of x 
            y_min (float):
                the min value of x
            y_max (float):
                the max value of x 
            valid_indicator (int):
                is used to determin if there is a valid point in a grid cell
    
    """

    def __init__(
        self,
        grid_size: int = 64,
        x_min: float = -1.0,
        x_max: float = 1.0,
        y_min: float = -1.0,
        y_max: float = 1.0,
        valid_indicator: float = 1.0
    ):
        self.grid_size = grid_size
        self.x_min, self.x_max, self.y_min, self.y_max = x_min, x_max, y_min, y_max
        self.valid_indicator = valid_indicator
        self.grid_step_x = (x_max - x_min) / grid_size
        self.grid_step_y = (y_max - y_min) / grid_size
        self._neighbors = GridNeighbors(self.grid_size, self.grid_size)


    def _points_to_grid(
        self, 
        point: np.array,
        point_grid_pos: np.array,
    ) -> np.array:
        """
            Takes one point and the assigned position in the grid and brings it in the format the grid expect it. 
            The grid expectes a format that is valid_indicator, x_div_center, y_dic_center, rest.
            
            Parameters:
                points (np.array):
                    a array one unscaled point  

            Returns:
                point (np.array):
                    point vlaues transformed for adding them into the grid
        """

        scaled_x = (point[:, 0] - self.x_min) / self.grid_step_x
        scaled_y = (point[:, 1] - self.y_min) / self.grid_step_y

        dx = scaled_x - (point_grid_pos[:, 0] + 0.5)
        dy = scaled_y - (point_grid_pos[:, 1] + 0.5)
        rest = point[:, 2:]

        values = np.hstack((
            np.full((point.shape[0], 1), self.valid_indicator),
            dx[:, None],
            dy[:, None],
            rest
        ))

        return values


    def points_to_grid(
        self,
        points: np.array
    ) -> np.array:
        """
            Here we assing all of the points given to our grid of the size self.grid_size x self.grid_size by using the first two 
            dims of the Points x,y to assing the corusponding cell in the grid. The cells in the grid get a corosponding x,y with
            the help of self.x_min, self.x_max, self.y_min and self.y_max. If two points fall in the same grid cell one of the points
            will be shifted to a near cell so that sum of distances from all points to the corosponding grid cell is minimal.
            The data in the grid is saved like this. Dim 0 is eather self.valid_indicator or 0, dim 1 is the distance from the original
            x of the point to the corospoinding x in the middel of the grid cell, dim 2 is the distance in y, the rest of the dims are dim [2:]
            of the original points  

            Parameters:
                points (np.array):
                    a array of all of the unscaled points       

            Returns:
                grid (np.array):
                    the points transformed to grid structure
        """
        
        # start_time = time.process_time()

        if not isinstance(points, np.ndarray):
            points = np.array(points)

        grid = np.zeros((self.grid_size, self.grid_size, 1 + points.shape[1]), dtype=np.float32)  # init the grid to be filled after

        # calculate the cord of the cell in the grid the points belong to
        grid_coords = np.clip(((points[:, : 2] - [self.x_min, self.y_min]) / [self.grid_step_x, self.grid_step_y]).astype(int), 
                              a_min=0, 
                              a_max=self.grid_size - 1)         # we need to make sure that we dont gett cells that are outside of the grid
        
        _, inverse, counts = np.unique(grid_coords, axis=0, return_inverse=True, return_counts=True)    # get all the uniqe points and the ones not unique by getting the nummber the same x,y accured and the inverse index of all points
        unique_mask = counts[inverse] == 1
        
        # handel all points that are alone in there cell in the grid
        unique_points = points[unique_mask]
        unique_points_cords = grid_coords[unique_mask]
        grid[unique_points_cords[:, 0], unique_points_cords[:, 1]] = self._points_to_grid(unique_points, unique_points_cords)    # we fill the grid with all of the different uniqe points 

        # after_cell_fill_time = time.process_time()

        # now we need to handel the duplicate grid cell entrys by finding a other grid cell for them minimizing the distance to the corosponding pixxel in the grid 
        duplicate_points = points[~unique_mask]
        duplicate_points_cords = grid_coords[~unique_mask]

        if sum(~unique_mask) > 0:   # we have points that fall into the same cell and we need to handel them 
            # first we need to find enoght valid close potential grid cells (the grid cell must be empty and we need to have more than we have points) for our points so later we our 
            # linear optimization problem has rather low complexity                                
            distance = 4                                         # here we save our distance of the gird cells we 
            candidate_cells = set()                             # here we save how many potential points we have
            while len(candidate_cells) < duplicate_points.shape[0]:
                potential_neighbors = self._neighbors.get_neighbors(duplicate_points_cords, distance)
                valid_cell_mask = grid[potential_neighbors[:, 0], potential_neighbors[:, 1], 0] != self.valid_indicator      # we need to check all the ptential grid cell entrys with radius if they are empty        
                candidate_cells.update(map(tuple, potential_neighbors[valid_cell_mask]))
                distance += 1

            candidate_cells = list(candidate_cells) 
            
            # after_neighborhood_time = time.process_time()

            # now we need to fill our cost matrix for our linear sum assignment problem 
            scaled_points = (duplicate_points[:, : 2] - [self.x_min, self.y_min]) / [self.grid_step_x, self.grid_step_y]
            centers = np.array(candidate_cells, dtype=float) + 0.5 

            dx = scaled_points[:, 0].reshape(-1, 1) - centers[:, 0].reshape(1, -1)   
            dy = scaled_points[:, 1].reshape(-1, 1) - centers[:, 1].reshape(1, -1)  
            cost_matrix = dx**2 + dy**2  

            # pre_linear_sum_assignment_time = time.process_time()

            row_ind, col_ind = linear_sum_assignment(cost_matrix)   # the cost matrix has the sape scaled_points.shape[0] x centers.shape[0] so number of points x posible cells
            
            # post_linear_sum_assignment_time = time.process_time()
            
            duplicate_points_new_cords = np.array(candidate_cells)[col_ind]
            duplicate_points = duplicate_points[row_ind]

            # now we have unique cords for all our points that had duplicate cells in the first place so we can assign them know
            grid[duplicate_points_new_cords[:, 0], duplicate_points_new_cords[: , 1]] = self._points_to_grid(duplicate_points, duplicate_points_new_cords) 
        
            # grid_filling_time = time.process_time()

        # print(f"Needed time: {grid_filling_time - start_time:.6f} - "+
        #      f"for cell filling: {after_cell_fill_time - start_time:.6f} - "+
        #      f"for neighborhood search: {after_neighborhood_time - after_cell_fill_time:.6f} - "+
        #      f"cost matrix creation: {pre_linear_sum_assignment_time - after_neighborhood_time:.6f} - "+
        #      f"for linear sum assignment: {post_linear_sum_assignment_time - pre_linear_sum_assignment_time:.6f} - "+
        #      f"for grid filling: {grid_filling_time - post_linear_sum_assignment_time:.6f}")

        return grid


    def point_from_gridentry(
        self, 
        grid: np.array, 
        row: int, 
        col: int
    ) -> np.array:
        """ 
            This function takes one grid entry cell where a valid point is and transforms it back into a point form
        """
        entry = grid[:, row, col]

        x_scaled = entry[1] + (row + 0.5)
        y_scaled = entry[2] + (col + 0.5)
        x_unscaled, y_unscaled = self.x_min + x_scaled * self.grid_step_x , self.y_min + y_scaled * self.grid_step_y

        return np.concatenate(([x_unscaled, y_unscaled], entry[3:]))
    
    
    def _grid_to_points_single(self, grid, valid_threshold):
        if valid_threshold is None:
            mask = (grid[0] == self.valid_indicator)
        else:
            mask = (grid[0] >= valid_threshold)

        row_idx, col_idx = np.where(mask)

        pts = [self.point_from_gridentry(grid, r, c) for r, c in zip(row_idx, col_idx)]
        return np.array(pts, dtype=np.float32)


    def grid_to_points(
        self,
        grid: np.array,
        valid_threshold: float = None
    ) -> np.array:
        """
            Here we get a grid of the size self.grid_size, grid_size where every entery is of the form valid_indicator, x_dist_to_center, y_dist_to_center, ....
            We get the points from the grid by going over the grid and for every cell where the valid_indicator == self.valid_indicator we take the center cordinate 
            of the grid cell and add the x_dist_to_center and y_dist_to_center and just ceap the last dimensions.
        """
        grid = np.asarray(grid)

        if grid.ndim == 3:            # shape = (C,H,W)
            return self._grid_to_points_single(grid, valid_threshold)
        elif grid.ndim == 4:      
            batch_points = []
            for g in grid:
                pts = self._grid_to_points_single(g, valid_threshold)
                batch_points.append(pts)
            return batch_points

        else:
            raise ValueError(f"Unsupported grid shape {grid.shape}. Expected (C,H,W) or (B,C,H,W).")


    def grid_to_image(
            self, 
            grid, 
            swap_xy=False, 
            invert_rows=False, 
            invert_columns=False
    ) -> np.array:
        """
            Radar grids have x, y in rows and columns, respectively.
            to display them as images alonside scatter plots, one may need to swap x and y
            and invert the order of columns and rows after swapping
        """
        
        if grid.dtype == np.float16:
                grid = grid.astype(np.float32)

        grid = np.swapaxes(grid, 0, 1) if swap_xy else grid

        row_step = -1 if invert_rows else 1
        col_step = -1 if invert_columns else 1

        return grid[::row_step, ::col_step, :]
