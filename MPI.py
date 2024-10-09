from mpi4py import MPI
import numpy as np
from Problem import Problem
from Method import Method
from Point import Point


# Constants for the simulation
ITERATIONS = 10  # Number of iterations for the simulation
MESH_SIZE = 1 / 20  # Size of the mesh
OMEGA = 0.8  # Relaxation factor (if needed)

if __name__ == "__main__":
    # Initialize the MPI communication
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()  # Get the rank of the process

    num_points_boundary = int(1 / MESH_SIZE)  # Number of boundary points based on mesh size

    # Initialize problem instances for each omega (area)
    problem1 = Problem(Point(0, 0), Point(1, 1), Point(1, 1), Point(0, 0), MESH_SIZE, MESH_SIZE, [])
    problem2 = Problem(Point(0, 0), Point(1, 2), Point(1, 2), Point(0, 0), MESH_SIZE, MESH_SIZE, [])
    problem3 = Problem(Point(0, 0), Point(1, 1), Point(1, 1), Point(0, 0), MESH_SIZE, MESH_SIZE, [])

    if rank == 0:
        # Set boundary conditions for problem1
        problem1.update_boundary(np.ones(num_points_boundary) * 40, 0)  # Left
        problem1.update_boundary(np.ones(num_points_boundary) * 15, 2)  # Top
        problem1.update_boundary(np.ones(num_points_boundary) * 15, 3)  # Bottom
        
        temp_results = []

        for i in range(ITERATIONS):
            # Send boundaries to problem2 (rank 1)
            comm.send(problem1.update_boundary(), dest=1)
            dirichlet_boundary = comm.recv(source=1)  # Receive updated boundaries from problem2
            
            # Update problem1 with the Dirichlet boundary
            problem1.update_boundary(dirichlet_boundary, 1)  # Assume '1' corresponds to the suitable wall

            # Solve problem1
            result = Method.solve(problem1)
            temp_results.append(result)

        comm.send(temp_results, dest=3)  # Send results to rank 3 for presentation

    elif rank == 1:
        # Set boundary conditions for problem2
        problem2.update_boundary(np.ones(num_points_boundary) * 40, 2)  # Top
        problem2.update_boundary(np.ones(num_points_boundary) * 5, 3)  # Bottom

        for i in range(ITERATIONS):
            # Solve problem2
            result = Method.solve(problem2)
            
            # Send left Dirichlet boundary to problem1
            comm.send(result, dest=0)
            # Receive Dirichlet boundary from problem1
            dirichlet_boundary = comm.recv(source=0)
            # Update the boundaries of problem2 with the received value
            problem2.update_boundary(dirichlet_boundary, 0)  # Update left boundary

            # Prepare right Dirichlet send to problem3 (rank 2)
            error_bound = np.ones(num_points_boundary)  # replace with your logic for error bounds
            comm.send(error_bound, dest=2)

    elif rank == 2:
        # Set boundary conditions for problem3
        problem3.update_boundary(np.ones(num_points_boundary) * 15, 2)  # Top
        problem3.update_boundary(np.ones(num_points_boundary) * 15, 3)  # Bottom
        problem3.update_boundary(np.ones(num_points_boundary) * 40, 4)  # Right

        for i in range(ITERATIONS):
            dirichlet_boundary_from_omega2 = comm.recv(source=1)  # Receive right boundary from problem2
            # Update boundaries of problem3
            problem3.update_boundary(dirichlet_boundary_from_omega2, 0)  

            # Solve problem3
            result = Method.solve(problem3)

            # Send Dirichlet boundary back to problem2
            comm.send(result, dest=1)

    elif rank == 3:
        # This rank presents the results
        results_from_omega1 = comm.recv(source=0)  # Receive results from problem1
        results_from_omega2 = comm.recv(source=1)  # Receive results from problem2
        results_from_omega3 = comm.recv(source=2)  # Receive results from problem3

        # Print results for each of the omega problems
        print("\nResults from Omega 1:")
        print(results_from_omega1)
        
        print("\nResults from Omega 2:")
        print(results_from_omega2)
        
        print("\nResults from Omega 3:")
        print(results_from_omega3)

