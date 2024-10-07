import numpy as np
import scipy as sp

class Method:
    def compute_b(problem):
        '''
        Should create a b matrix with the RHS of the discretized equation 

        Parameters:
            new_b: Tuple, should specify which bs were changed corresponding to the Neumann, Dirichlet condition 

        for the inside values RHS = 0
        for boundary_values RHS = Gamma_H/Gamma_WF/Gamma_W
        for the unknown wall RHS = new_b (needs to be activley changed!)
        '''
        return b

    def compute_A(problem):
        '''
        Should create a A matrix with the LHS of the discretized equation

        Parameters:
            boundary_condition: Tuple, should specify which bs correspond to which condition (Neumann, Dirichlet)

        Should use self.boundary_values to set the LHS for the unknown points on the wall Gamma_i depending on the condition
        Inner points should follow the approximation equation from the lecture
        For known boundary points set the point in the matrix A = 1 (no equation to compute needed as we already have the point given)
        '''
        return A

    def solve(self, problem):
        A = self.compute_A(problem)
        b = self.compute_b(problem)
        return sp.linalg.solve(A,b)

