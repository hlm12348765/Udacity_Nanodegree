import math
from math import sqrt
import numbers

def zeroes(height, width):
        """
        Creates a matrix of zeroes.
        """
        g = [[0.0 for _ in range(width)] for __ in range(height)]
        return Matrix(g)

def identity(n):
        """
        Creates a n x n identity matrix.
        """
        I = zeroes(n, n)
        for i in range(n):
            I.g[i][i] = 1.0
        return I

def get_row(matrix, row):
    return matrix[row]

def get_column(matrix, column_number):
    column = []
    for i in range(len(matrix)):
        col = matrix[i][column_number]
        column.append(col)
    return column

def dot_product(vector_one, vector_two):
    result = 0
    re = 0
    re_sum = []
    if len(vector_one) != len(vector_two):
        print('These two vectors must have the same length!')
    for i in range(len(vector_one)):
        re = vector_one[i] * vector_two[i]
        re_sum.append(re)
    result = sum(re_sum)
    return result

class Matrix(object):

    # Constructor
    def __init__(self, grid):
        self.g = grid
        self.h = len(grid)
        self.w = len(grid[0])

    #
    # Primary matrix math methods
    #############################
 
    def determinant(self):
        """
        Calculates the determinant of a 1x1 or 2x2 matrix.
        """
        if not self.is_square():
            raise ValueError("Cannot calculate determinant of non-square matrix.")

        det = 0
        if self.h == 1:
            det = self.g[0][0]
        else:
            for i in range(self.h):
                MC = [[row[j] for j in range(self.h) if j != i] for row in self.g[1:]]
                other = Matrix(MC)
                det += self.g[0][i] * Matrix.determinant(other) * (-1)**(i)
        return det

    def trace(self):
        """
        Calculates the trace of a matrix (sum of diagonal entries).
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate the trace of a non-square matrix.")

        m_trace = 0
        for i in range(self.h):
            for j in range(self.w):
                if j == i:
                    m_trace += self.g[i][j]
        return m_trace

    def inverse(self):
        """
        Calculates the inverse of a 1x1 or 2x2 Matrix.
        """
        if not self.is_square():
            raise(ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise(NotImplementedError, "inversion not implemented for matrices larger than 2x2.")

        if self.h == 1:
            inverse = Matrix([[]])
            inverse[0].append(1/self.g[0][0])
        elif self.h == 2:
            inverse = Matrix([[], []])
            det = Matrix.determinant(self)
            if det == 0:
                raise ValueError('The matrix is not invertible')
            else:
                for j in range(2):
                    for i in range(2):
                        inverse[j].append((1/det) * (-1)**(2-i+j)*self.g[1-i][1-j])
  
        return inverse

    def T(self):
        """
        Returns a transposed copy of this Matrix.
        """
        matrix_transpose = []
        row = []
        for j in range(self.w):
            for i in range(self.h):
                row.append(self.g[i][j])
            matrix_transpose.append(row)
            row = []
        MT = Matrix(matrix_transpose)
        return MT

    def is_square(self):
        return self.h == self.w

    #
    # Begin Operator Overloading
    ############################
    def __getitem__(self,idx):
        """
        Defines the behavior of using square brackets [] on instances
        of this class.

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > my_matrix[0]
          [1, 2]

        > my_matrix[0][0]
          1
        """
        return self.g[idx]

    def __repr__(self):
        """
        #Defines the behavior of calling print on an instance of this class.
        """
        s = ""
        for row in self.g:
            s += " ".join(["{} ".format(x) for x in row])
            s += "\n"
        return s

    def __add__(self,other):
        """
        Defines the behavior of the + operator
        """
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be added if the dimensions are the same") 

        matrixSum = []
        row = []
        if isinstance(self.g[0], list) == True and isinstance(other.g[0], list) == True:
            for i in range(self.h):
                for j in range(self.w):
                    result = self.g[i][j] + other.g[i][j]
                    row.append(result)
                matrixSum.append(row)
                row = []
        elif isinstance(self.g[0], list) == False or isinstance(other.g[0], list) == False:
            for i in range(self.h):
                result = self.g[i] + other.g[i]
                matrixSum.append(result)
        MSum = Matrix(matrixSum)
        return MSum

    def __neg__(self):
        """
        Defines the behavior of - operator (NOT subtraction)

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > negative  = -my_matrix
        > print(negative)
          -1.0  -2.0
          -3.0  -4.0
        """
        matrixNeg = []
        for i in range(self.h):
            matrixNeg.append([])
            for j in range(self.w):
                matrixNeg[i].append(-self.g[i][j])
        MN = Matrix(matrixNeg)
        return MN

    def __sub__(self, other):
        """
        Defines the behavior of - operator (as subtraction)
        """
        matrixSub = []
        row = []
        for i in range(self.h):
            for j in range(self.w):
                result = other.g[i][j] - self.g[i][j]
                row.append(result)
            matrixSub.append(row)
            row = []
        MSub = Matrix(matrixSub)
        return  MSub

    def __mul__(self, other):
        """
        Defines the behavior of * operator (matrix multiplication)
        """
        result = []
        row_result = []
        for i in range(self.h):
            for j in range(other.w):
                row_A = get_row(self.g, i)
                column_B = get_column(other.g, j)
                dot_p = dot_product(row_A, column_B)
                row_result.append(dot_p)
            result.append(row_result)
            row_result = []
        mulM = Matrix(result)
        return mulM

    def __rmul__(self, other):
        """
        Called when the thing on the left of the * is not a matrix.

        Example:

        > identity = Matrix([ [1,0], [0,1] ])
        > doubled  = 2 * identity
        > print(doubled)
          2.0  0.0
          0.0  2.0
        """
        if isinstance(other, numbers.Number):
            rmul = []
            for i in range(self.h):
                rmul.append([])
                for j in range(self.w):
                    rmul[i].append(other * self.g[i][j])
        rmulM = Matrix(rmul)
        return rmulM