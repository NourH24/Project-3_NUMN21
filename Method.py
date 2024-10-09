import numpy as np
import scipy as sp

class Method:
    def compute_b(self, problem):
        """
        Creates a sparse vector b representing the RHS of the discretized equation.
        
        For inside points, RHS = 0.
        For boundary values, RHS is determined by boundary conditions:
            - Gamma_H/Gamma_WF/Gamma_W for known boundaries
            - Gamma_i for the unknown boundary wall
            
        Parameters:
            problem (Problem): takes the parameters from the problem class

        Returns:
            b: A scipy sparse vector
        """

        # Get the number of grid points based on delta_x and delta_y
        self.nx = int((problem.B.x - problem.A.x) / problem.delta_x) + 1  # plus one to account for the one missing element in this
        self.ny = int((problem.D.y - problem.A.y) / problem.delta_y) + 1
        n_total = self.nx * self.ny  # Total number of points in the grid
        self.h_x = problem.delta_x
        self.h_y = problem.delta_y
        
        # Initialize b as a 2D array with zeros
        self.data_b = []
        self.rows_b = []
        self.cols_b = []
        
        # Computing all the different b's for the different boundaries + conditions
        self.boundary_lower_b(problem)
        self.boundary_right_b(problem)
        self.boundary_upper_b(problem)
        self.boundary_left_b(problem) 

         # Create the sparse RHS vector
        b_sparse = sp.csr_matrix((self.data_b, (self.rows_b, self.cols_b)), shape=(n_total, 1))

        return b_sparse

    def compute_A(self, problem):
        '''
        Should create a A matrix with the LHS of the discretized equation

        Parameters:
            boundary_condition: Tuple, should specify which bs correspond to which condition (Neumann, Dirichlet)

        Should use self.boundary_values to set the LHS for the unknown points on the wall Gamma_i depending on the condition
        Inner points should follow the approximation equation from the lecture
        For known boundary points set the point in the matrix A = 1 (no equation to compute needed as we already have the point given)
        '''
        nx = int((problem.B.x - problem.A.x) / problem.delta_x) + 1
        ny = int((problem.D.y - problem.A.y) / problem.delta_y) + 1

        self.h_x_A = problem.delta_x
        self.h_y_A = problem.delta_y
        
        self.rows_A = []
        self.cols_A = []
        self.data_A = []
        
        self.interior_A(problem)
        self.boundary_lower_A(problem)
        self.boundary_left_A(problem)
        self.boundary_upper_A(problem)
        self.boundary_right_A(problem)

        # Create sparse matrix A in csr format
        A = sp.csr_matrix((self.data_A, (self.rows_A, self.cols_A)), shape=(nx*ny, nx*ny))
        
        # Return the sparse matrix
        return A

        

    def solve(self, problem):
        A = self.compute_A(problem)
        b = self.compute_b(problem)
        return sp.linalg.solve(A,b)


    def boundary_lower_b(self, problem):

        boundary_length = problem.B.x - problem.A.x  # Total length of the lower boundary
        k = 0
        for condition_type, value, length in problem.boundary_conditions[0]:
            # Convert length to number of grid points
            num_grid_points = int((length / boundary_length) * self.nx)
            
            if condition_type is 'Dirichlet':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.nx-1:
                        k += 1
                        continue
                    row_index = k + self.nx # Linear index for the lower boundary
                    self.data_b.append(value/(self.h_y**2))
                    self.rows_b.append(row_index)
                    self.cols_b.append(0)
                    k += 1
                    # What happens to the boundary points
            elif condition_type is 'Neumann':
                for i in range(num_grid_points):
                    if k == 0:
                        self.bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.nx-1:
                        self.bottom_right_c = value
                        k += 1
                        continue
                row_index = k  # Linear index for the lower boundary
                self.data_b.append(value/self.h_y)
                self.rows_b.append(row_index)
                self.cols_b.append(0)
                k += 1
            
            else:
                for i in range(num_grid_points):
                    if k == 0:
                        bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.nx-1:
                        bottom_right_c = value
                        k += 1
                        continue

                    row_index = k  # Linear index for the lower boundary
                    self.data_b.append(value)
                    self.rows_b.append(row_index)
                    self.cols_b.append(0)
                    k += 1

        assert k == self.nx, "k should be the same as nx, indexing error"


    def boundary_right_b(self, problem):

        boundary_length = problem.C.x - problem.B.x  # Total length of the lower boundary
        k = 0
        for condition_type, value, length in problem.boundary_conditions[1]:
            # Convert length to number of grid points
            num_grid_points = int((length / boundary_length) * self.ny)
            
            if condition_type is 'Dirichlet':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.ny-1:
                        k += 1
                        continue
                    row_index = (k + 1) * self.nx - 2 # Linear index for the lower boundary
                    self.data_b.append(value/(self.h_x**2))
                    self.rows_b.append(row_index)
                    self.cols_b.append(0)
                    k += 1
                    # What happens to the boundary points

            elif condition_type is 'Neumann':
                for i in range(num_grid_points):
                    if k == 0:
                        self.bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.ny-1:
                        self.bottom_right_c = value
                        k += 1
                        continue
                row_index = (k + 1) * self.nx - 1 # Linear index for the lower boundary
                self.data_b.append(value/self.h_x)
                self.rows_b.append(row_index)
                self.cols_b.append(0)
                k += 1
            
            else:
                for i in range(num_grid_points):
                    if k == 0:
                        bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.ny-1:
                        bottom_right_c = value
                        k += 1
                        continue

                    row_index = (k + 1) * self.nx - 1 # Linear index for the lower boundary
                    self.data_b.append(value)
                    self.rows_b.append(row_index)
                    self.cols_b.append(0)
                    k += 1

        assert k == self.ny, "k should be the same as nx, indexing error"


    def boundary_upper_b(self, problem):

        boundary_length = problem.C.x - problem.D.x  # Total length of the lower boundary
        k = 0
        for condition_type, value, length in problem.boundary_conditions[2]:
            # Convert length to number of grid points
            num_grid_points = int((length / boundary_length) * self.nx)
            
            if condition_type is 'Dirichlet':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.nx-1:
                        k += 1
                        continue
                    row_index = (self.ny - 2) * self.nx + k
                    self.data_b.append(value/(self.h_y**2))
                    self.rows_b.append(row_index)
                    self.cols_b.append(0)
                    k += 1
                    # What happens to the boundary points
            elif condition_type is 'Neumann':
                for i in range(num_grid_points):
                    if k == 0:
                        self.bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.nx-1:
                        self.bottom_right_c = value
                        k += 1
                        continue
                row_index = (self.ny - 1) * self.nx + k
                self.data_b.append(value/self.h_y)
                self.rows_b.append(row_index)
                self.cols_b.append(0)
                k += 1
            
            else:
                for i in range(num_grid_points):
                    if k == 0:
                        bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.nx-1:
                        bottom_right_c = value
                        k += 1
                        continue

                    row_index = (self.ny - 1) * self.nx + k
                    self.data_b.append(value)
                    self.rows_b.append(row_index)
                    self.cols_b.append(0)
                    k += 1

        assert k == self.nx, "k should be the same as nx, indexing error"


    def boundary_left_b(self, problem):

        boundary_length = problem.D.x - problem.A.x  # Total length of the lower boundary
        k = 0
        for condition_type, value, length in problem.boundary_conditions[3]:
            # Convert length to number of grid points
            num_grid_points = int((length / boundary_length) * self.ny)
            
            if condition_type is 'Dirichlet':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.ny-1:
                        k += 1
                        continue
                    row_index = k *self.nx + 1
                    self.data_b.append(value/(self.h_x**2))
                    self.rows_b.append(row_index)
                    self.cols_b.append(0)
                    k += 1
                    # What happens to the boundary points

            elif condition_type is 'Neumann':
                for i in range(num_grid_points):
                    if k == 0:
                        self.bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.ny-1:
                        self.bottom_right_c = value
                        k += 1
                        continue
                row_index = k* self.nx
                self.data_b.append(value/self.h_x)
                self.rows_b.append(row_index)
                self.cols_b.append(0)
                k += 1
            
            else:
                for i in range(num_grid_points):
                    if k == 0:
                        bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.ny-1:
                        bottom_right_c = value
                        k += 1
                        continue

                    row_index = k * self.nx - 1 # Linear index for the lower boundary
                    self.data_b.append(value)
                    self.rows_b.append(row_index)
                    self.cols_b.append(0)
                    k += 1

        assert k == self.ny, "k should be the same as nx, indexing error"


    def interior_A(self, problem):
        # Iterate over interior points
        for i in range(2, self.ny - 2):  # from 1 to ny - 2
            for j in range(2, self.nx - 2):  # from 1 to nx - 2
                # Calculate the linear index for interior point U_{i,j}
                linear_index = i * self.ny + j

                # Assign coefficients based on the finite difference equation
                # Center point (v_{i,j})
                self.rows_A.append(linear_index)
                self.cols_A.append(linear_index)  
                self.data_A.append(-2 / self.h_x_A**2 - 2 / self.h_y_A**2)  

                # Left neighbor (v_{i-1,j})
                self.rows_A.append(linear_index)
                self.cols_A.append(linear_index - 1)  
                self.data_A.append(1 / self.h_x_A**2)  

                # Right neighbor (v_{i+1,j})
                self.rows_A.append(linear_index)
                self.cols_A.append(linear_index + 1)  
                self.data_A.append(1 / self.h_x_A**2) 

                # Bottom neighbor (v_{i,j-1})
                self.rows_A.append(linear_index)
                self.cols_A.append(linear_index - self.nx) 
                self.data_A.append(1 / self.h_y_A**2)  

                # Top neighbor (v_{i,j+1})
                self.rows_A.append(linear_index)
                self.cols_A.append(linear_index + self.nx )  
                self.data_A.append(1 / self.h_y_A**2)


    def boundary_lower_A(self, problem):

        boundary_length = problem.B.x - problem.A.x  # Total length of the lower boundary
        k = 0
        for condition_type, value, length in problem.boundary_conditions[0]:
            # Convert length to number of grid points
            num_grid_points = int((length / boundary_length) * self.nx)
            
            if condition_type is 'Dirichlet':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.nx-1:
                        k += 1
                        continue
                    row_index = k + self.nx # Linear index for the lower boundary

                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index)  
                    self.data_A.append(2 / self.h_y_A**2 + 2 / self.h_x_A**2)  

                    # Left neighbor (v_{i-1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - 1)  
                    self.data_A.append(-1 / self.h_x_A**2)  

                    # Right neighbor (v_{i+1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + 1)  
                    self.data_A.append(-1 / self.h_x_A**2) 

                    # Top neighbor (v_{i,j+1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + self.nx )  
                    self.data_A.append(-1 / self.h_y_A**2)

                    k += 1
                    # What happens to the boundary points
            elif condition_type is 'Neumann':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.nx-1:
                        k += 1
                        continue
                    row_index = k  # Linear index for the lower boundary
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index)  
                    self.data_A.append(- 1 / self.h_y_A**2 - 2 / self.h_x_A**2)  

                    # Left neighbor (v_{i-1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - 1)  
                    self.data_A.append(1 / self.h_x_A**2)  

                    # Right neighbor (v_{i+1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + 1)  
                    self.data_A.append(1 / self.h_x_A**2) 

                    # Top neighbor (v_{i,j+1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + self.nx )  
                    self.data_A.append(1 / self.h_y_A**2)

                    k += 1
            
            else:
                for i in range(num_grid_points):
                    if k == 0:
                        bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.nx-1:
                        bottom_right_c = value
                        k += 1
                        continue

                    row_index = k  # Linear index for the lower boundary
                    self.data_b.append(1)
                    self.rows_b.append(row_index)
                    self.cols_b.append(row_index)
                    k += 1

        assert k == self.nx, "k should be the same as nx, indexing error"


    def boundary_upper_A(self, problem):

        boundary_length = problem.C.x - problem.D.x  # Total length of the lower boundary
        k = 0
        for condition_type, value, length in problem.boundary_conditions[2]:
            # Convert length to number of grid points
            num_grid_points = int((length / boundary_length) * self.nx)
            
            if condition_type is 'Dirichlet':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.nx-1:
                        k += 1
                        continue
                    row_index =  (self.ny - 2) * self.nx + k

                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index)  
                    self.data_A.append(2 / self.h_y_A**2 + 2 / self.h_x_A**2)  

                    # Left neighbor (v_{i-1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - 1)  
                    self.data_A.append(-1 / self.h_x_A**2)  

                    # Right neighbor (v_{i+1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + 1)  
                    self.data_A.append(-1 / self.h_x_A**2) 

                    # Bottom neighbor (v_{i,j-1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - self.nx )  
                    self.data_A.append(-1 / self.h_y_A**2)

                    k += 1
                    # What happens to the boundary points
            elif condition_type is 'Neumann':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.nx-1:
                        k += 1
                        continue
                    row_index = (self.ny - 1) * self.nx + k
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index)  
                    self.data_A.append(1 / self.h_y_A**2 + 2 / self.h_x_A**2)  

                    # Left neighbor (v_{i-1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - 1)  
                    self.data_A.append(-1 / self.h_x_A**2)  

                    # Right neighbor (v_{i+1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + 1)  
                    self.data_A.append(-1 / self.h_x_A**2) 

                    # Bottom neighbor (v_{i,j-1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - self.nx )  
                    self.data_A.append(-1 / self.h_y_A**2)

                    k += 1
            
            else:
                for i in range(num_grid_points):
                    if k == 0:
                        bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.nx-1:
                        bottom_right_c = value
                        k += 1
                        continue

                    row_index = (self.ny - 1) * self.nx + k
                    self.data_b.append(1)
                    self.rows_b.append(row_index)
                    self.cols_b.append(row_index)
                    k += 1

        assert k == self.nx, "k should be the same as nx, indexing error"

    def boundary_right_A(self, problem):

        boundary_length = problem.C.x - problem.B.x  # Total length of the lower boundary
        k = 0
        for condition_type, value, length in problem.boundary_conditions[1]:
            # Convert length to number of grid points
            num_grid_points = int((length / boundary_length) * self.nx)
            
            if condition_type is 'Dirichlet':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.nx-1:
                        k += 1
                        continue
                    row_index =  (k + 1) * self.nx - 2

                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index)  
                    self.data_A.append(2 / self.h_y_A**2 + 2 / self.h_x_A**2)  

                    # Left neighbor (v_{i-1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - 1)  
                    self.data_A.append(-1 / self.h_x_A**2)  

                    # Upper neighbor (v_{i,j+1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + self.nx)  
                    self.data_A.append(-1 / self.h_y_A**2) 

                    # Bottom neighbor (v_{i,j-1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - self.nx )  
                    self.data_A.append(-1 / self.h_y_A**2)

                    k += 1
                    # What happens to the boundary points
            elif condition_type is 'Neumann':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.nx-1:
                        k += 1
                        continue
                    row_index = (k + 1) * self.nx - 1
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index)  
                    self.data_A.append(2 / self.h_y_A**2 + 1 / self.h_x_A**2)  

                    # Left neighbor (v_{i-1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - 1)  
                    self.data_A.append(-1 / self.h_x_A**2)  

                    # Upper neighbor (v_{i,j+1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + self.nx)  
                    self.data_A.append(-1 / self.h_y_A**2)

                    # Bottom neighbor (v_{i,j-1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - self.nx )  
                    self.data_A.append(-1 / self.h_y_A**2)

                    k += 1
            
            else:
                for i in range(num_grid_points):
                    if k == 0:
                        bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.nx-1:
                        bottom_right_c = value
                        k += 1
                        continue

                    row_index = (k + 1) * self.nx - 1
                    self.data_A.append(1)
                    self.rows_b.append(row_index)
                    self.cols_b.append(row_index)
                    k += 1

        assert k == self.nx, "k should be the same as nx, indexing error"


    def boundary_left_A(self, problem):

        boundary_length = problem.D.x - problem.A.x  # Total length of the lower boundary
        k = 0
        for condition_type, value, length in problem.boundary_conditions[3]:
            # Convert length to number of grid points
            num_grid_points = int((length / boundary_length) * self.nx)
            
            if condition_type is 'Dirichlet':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.nx-1:
                        k += 1
                        continue
                    row_index =  k *self.nx + 1

                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index)  
                    self.data_A.append(2 / self.h_y_A**2 + 2 / self.h_x_A**2)  

                    # Right neighbor (v_{i+1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + 1)  
                    self.data_A.append(-1 / self.h_x_A**2)  

                    # Upper neighbor (v_{i,j+1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + self.nx)  
                    self.data_A.append(-1 / self.h_y_A**2) 

                    # Bottom neighbor (v_{i,j-1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - self.nx )  
                    self.data_A.append(-1 / self.h_y_A**2)

                    k += 1
                    # What happens to the boundary points
            elif condition_type is 'Neumann':
                for i in range(num_grid_points):
                    if k == 0:
                        k += 1
                        continue
                    elif k == self.nx-1:
                        k += 1
                        continue
                    row_index = k* self.nx
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index)  
                    self.data_A.append(-2 / self.h_y_A**2 - 1 / self.h_x_A**2)  

                    # Right neighbor (v_{i+1,j})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + 1)  
                    self.data_A.append(1 / self.h_x_A**2)    

                    # Upper neighbor (v_{i,j+1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index + self.nx)  
                    self.data_A.append(1 / self.h_y_A**2)

                    # Bottom neighbor (v_{i,j-1})
                    self.rows_A.append(row_index)
                    self.cols_A.append(row_index - self.nx )  
                    self.data_A.append(1 / self.h_y_A**2)

                    k += 1
            
            else:
                for i in range(num_grid_points):
                    if k == 0:
                        bottom_left_c = value
                        k += 1
                        continue
                    elif k == self.nx-1:
                        bottom_right_c = value
                        k += 1
                        continue

                    row_index = k* self.nx
                    self.data_A.append(1)
                    self.rows_b.append(row_index)
                    self.cols_b.append(row_index)
                    k += 1

        assert k == self.nx, "k should be the same as nx, indexing error"