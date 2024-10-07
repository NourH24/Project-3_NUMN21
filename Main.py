#Main class
#MPI
import numpy as np
from Problem import Problem
from Method import Method

if _name_ == '_main_':
    mesh_size = 1/20
    omega = 0.8
    itterations = 10

    num_points_boundary = 1/mesh_size # Correct?

    problem1 = Problem(1, 1, mesh_size, mesh_size)
    problem2 = Problem(1, 2, mesh_size, mesh_size)
    problem3 = Problem(1, 1, mesh_size, mesh_size)

    # Check index of walls
    # Check what to init boundary walls with
    problem1.update_boundary(np.ones(num_points_boundary) * 40, 0) # Left
    problem1.update_boundary(np.ones(num_points_boundary) * 15, 2) # Top
    problem1.update_boundary(np.ones(num_points_boundary) * 15, 3) # Bottom
    
    problem2.update_boundary(np.ones(num_points_boundary) * 40, 2) # Top
    problem2.update_boundary(np.ones(num_points_boundary) * 5, 3) # Bottom

    problem3.update_boundary(np.ones(num_points_boundary) * 15, 2) # Top
    problem3.update_boundary(np.ones(num_points_boundary) * 15, 3) # Bottom
    problem3.update_boundary(np.ones(num_points_boundary) * 40, 4) # Right

    for i in range(itterations):
        u2 = Method.solve(problem2)
        
        # Update 2, 3...

        u1 = Method.solve(problem1)
        u3 = Method.solve(problem3)
