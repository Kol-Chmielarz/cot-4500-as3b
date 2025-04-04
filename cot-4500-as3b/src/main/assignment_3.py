import numpy as np

def gaussian_elim(A, b):
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    n = len(b)

    for i in range(n):
        column_slice = Ab[i:, i]  
        abs_values = np.abs(column_slice)   
        max_index = np.argmax(abs_values)
        max_row = i + max_index
        Ab[[i, max_row]] = Ab[[max_row, i]]
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    bsub = np.zeros(n)
    for i in range(n-1, -1, -1):
        right = Ab[i, -1]
        dp = np.dot(Ab[i, i+1:n], bsub[i+1:n])
        bsub[i] = (right - dp) / Ab[i, i]

    return bsub



def lu_fact(A):
    A = A.astype(float)
    n = A.shape[0]
    L = np.zeros((n, n))  
    for i in range(n):
        L[i, i] = 1.0  
    U = A.copy()

    for i in range(n):
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            U[j, i:] -= factor * U[i, i:]
            L[j, i] = factor

    det = np.prod(np.diag(U)) 
    return L, U, det


def diag_dom(A):
    n = A.shape[0]
    dom =  False

    for i in range(n):
        diag = abs(A[i, i])
        sum = 0
        for j in range(n):
            if j!=i:
                sum += abs(A[i, j])

        if diag < sum:
            dom = False
            return dom
        
        if diag > sum:
            dom = True
            

    return dom

def pos_def(A):
    n = A.shape[0]

    for i in range(n):
        for j in range(n):
            if A[i, j] != A[j, i]:
                return False
            

    A = A.astype(float)
    for i in range(n):
        pivot = A[i, i]
        if pivot <= 0:
            return False

        for j in range(i + 1, n):
            factor = A[j, i] / pivot
            A[j, i:] -= factor * A[i, i:]
    return True

