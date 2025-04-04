import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.main.assignment_3 import gaussian_elim
from src.main.assignment_3 import gaussian_elim, lu_fact
from src.main.assignment_3 import diag_dom
from src.main.assignment_3 import pos_def


def test_gaus_elim():
    A = np.array([
        [2, -1, 1],
        [1,  3, 1],
        [-1, 5, 4] 
        ])
    b = np.array([6, 0, -3])

    ans = gaussian_elim(A, b)

    print(ans)


def test_LU():
    A = np.array([
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ])

    L, U, det = lu_fact(A)
    print("L matrix:")
    print(L)
    print("\nU matrix:")
    print(U)
    print("\nDeterminant of A:", det)

def test_diag_dom():
    A = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ])
#Test matrix that is diag dominant
    #A = np.array([
    #[10, 1, 2, 0, 0],
    #[2, 11, 3, 1, 0],
    #[1, 1, 12, 2, 2],
    #[0, 2, 1, 13, 3],
    #[0, 0, 2, 1, 14]
#])
    
    result = diag_dom(A)

    if result == True:
        print("\nIs diagonally dominant\n")

    if result == False:
        print("\nIs not diagonally dominant\n")


def test_pos_def():
    A = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ])

#test not pos def matrix
#    A = np.array([
#    [0, 1, 1],
#    [1, 0, 1],
#    [1, 1, 0]
#])



    result = pos_def(A)

    if result == True:
        print("Is Positive Definite")

    if result == False:
        print("Is not Positive Definite")





if __name__ == "__main__":
    test_gaus_elim()
    test_LU()
    test_diag_dom()
    test_pos_def()
