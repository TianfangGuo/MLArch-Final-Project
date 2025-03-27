# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def inner_product(A, B):
    m = len(A)         # number of rows in A
    p = len(A[0])      # common dimension
    n = len(B[0])      # number of columns in B
    totalOPs = 0;
    
    # Initialize result matrix with zeros.
    C = [[0 for _ in range(n)] for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            total = 0
            for k in range(p):
                total += A[i][k] * B[k][j]
                totalOPs += 1;
            C[i][j] = total
            
    print("inner product number of operations: ", totalOPs)
    return C

def outer_product(A, B):
    m = len(A)
    p = len(A[0])
    n = len(B[0])
    totalOPs = 0;
    
    # Initialize result matrix with zeros.
    C = [[0 for _ in range(n)] for _ in range(m)]
    
    for k in range(p):
        # For each k, update the entire matrix with the contribution of the k-th column of A
        # and the k-th row of B.
        for i in range(m):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
                totalOPs += 1;
    print("outer product number of operations: ", totalOPs)
    return C

def gustavson(A, B):
    m = len(A)
    p = len(A[0])
    n = len(B[0])
    totalOPs = 0;
    
    # Initialize result matrix with zeros.
    C = [[0 for _ in range(n)] for _ in range(m)]
    
    for i in range(m):
        # Accumulator for the current row
        row_acc = [0 for _ in range(n)]
        for k in range(p):
            if A[i][k] != 0:  # in a sparse matrix, skip when the entry is zero
                for j in range(n):
                    row_acc[j] += A[i][k] * B[k][j]
                    totalOPs += 1;
        C[i] = row_acc
    print("gustavson's number of operations: ", totalOPs)
    return C

def csr_gustavson(csrA, csrB):
    dataA, indicesA, indptrA, shapeA = csrA
    dataB, indicesB, indptrB, shapeB = csrB
    m, kA = shapeA
    kB, n = shapeB
    totalOPs = 0;
    if kA != kB:
        raise ValueError("Inner dimensions do not match for multiplication.")
    
    C = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        row_acc = [0] * n
        # For every nonzero A[i,k] in row i of A:
        for a_idx in range(indptrA[i], indptrA[i+1]):
            k = indicesA[a_idx]
            a_val = dataA[a_idx]
            totalOPs += 2;
            # Get row k of B from its CSR structure.
            for b_idx in range(indptrB[k], indptrB[k+1]):
                j = indicesB[b_idx]
                b_val = dataB[b_idx]
                row_acc[j] += a_val * b_val
                totalOPs += 3;
        C[i] = row_acc
    print("gustavson's with CSR input number of operations: ", totalOPs)
    return C


def print_matrix(mat, name="Matrix"):
    print(f"{name}:")
    for row in mat:
        print("  ", row)
    print()
    
def dense_to_csr(matrix):
    data = []
    indices = []
    indptr = [0]  # Start of first row is at index 0 in data
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    for row in matrix:
        for j, value in enumerate(row):
            if value != 0:
                data.append(value)
                indices.append(j)
        # After processing a row, record the current length of data as the end pointer for that row.
        indptr.append(len(data))
    
    return data, indices, indptr, (m, n)

def print_csr(data, indices, indptr):
    print("CSR Representation:")
    print("data   =", data)
    print("indices=", indices)
    print("indptr =", indptr)
    print()

def main():
    A = [
        [1, 0],
        [3, 5]
    ]
    B = [
        [6, 7],
        [0, 8]
    ]
    
    C = [
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
        [8, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0],
        [0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    D = [
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 8, 0], 
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 9, 0, 0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0, 2, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
        [0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
        [7, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]


    C_inner = inner_product(C, D)
    #print_matrix(C_inner, "Result using Inner Product Dataflow")

    C_outer = outer_product(C, D)
    #print_matrix(C_outer, "Result using Outer Product Dataflow")

    C_gustavson = gustavson(C, D)
    #print_matrix(C_gustavson, "Result using Gustavson's Algorithm")

    csrC = dense_to_csr(C)
    csrD = dense_to_csr(D)
    #print_csr(data, indices, indptr)
    
    C_csr_gustavson = csr_gustavson(csrC, csrD)
    
    same = (C_inner == C_outer == C_gustavson)
    print("All methods produce the same result:", same)
    same2 = C_gustavson == C_csr_gustavson
    print("Gustavson CSR format matches naive format output: ", same2)
    

if __name__ == '__main__':
    main()
