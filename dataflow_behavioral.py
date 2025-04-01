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
                totalOPs += 1;
                for j in range(n):
                    row_acc[j] += A[i][k] * B[k][j]
                    totalOPs += 1;
        C[i] = row_acc
    print("gustavson algo number of operations: ", totalOPs)
    return C

def inner_product_csr_csc(csrA, cscB):
    dataA, indicesA, indptrA, shapeA, input_conversion_overheadA = csrA
    dataB, indicesB, indptrB, shapeB, input_conversion_overheadB = cscB
    conversion_overhead = input_conversion_overheadA + input_conversion_overheadB
    totalOPs = 0
    m, kA = shapeA
    kB, n = shapeB

    if kA != kB:
        raise ValueError("Inner dimensions do not match for multiplication.")
    
    # Initialize the result matrix as a dense array
    C = [[0 for _ in range(n)] for _ in range(m)]
    
    # For each row of A (in CSR format)
    for i in range(m):
        row_start = indptrA[i]
        row_end = indptrA[i + 1]
        row_indices = indicesA[row_start:row_end]
        row_values = dataA[row_start:row_end]
        
        # For each column of B (in CSC format)
        for j in range(n):
            col_start = indptrB[j]
            col_end = indptrB[j + 1]
            col_indices = indicesB[col_start:col_end]
            col_values = dataB[col_start:col_end]
            
            # Compute the dot product of A[i,:] and B[:,j]
            dot = 0
            a_ptr, b_ptr = 0, 0
            while a_ptr < len(row_indices) and b_ptr < len(col_indices):
                a_index = row_indices[a_ptr]
                b_index = col_indices[b_ptr]
                totalOPs += 1;
                if a_index == b_index:
                    dot += row_values[a_ptr] * col_values[b_ptr]
                    a_ptr += 1
                    b_ptr += 1
                    totalOPs +=1;
                elif a_index < b_index:
                    a_ptr += 1
                else:
                    b_ptr += 1
            
            C[i][j] = dot
    
    print("inner product with CSRxCSC inputs number of operations: ", totalOPs+conversion_overhead, " (", totalOPs, "+", conversion_overhead, ")")
    return C

def outer_product_csc_csr(cscA, csrB):
    dataA, indicesA, indptrA, shapeA, input_conversion_overheadA = cscA
    dataB, indicesB, indptrB, shapeB, input_conversion_overheadB = csrB
    conversion_overhead = input_conversion_overheadA + input_conversion_overheadB
    totalOPs = 0
    m, kA = shapeA      # A is of size m x kA.
    kB, n = shapeB      # B is of size kB x n.
    if kA != kB:
        raise ValueError("Inner dimensions do not match for multiplication.")
    
    # Allocate result dense matrix C (size m x n) with zeros.
    C = [[0 for _ in range(n)] for _ in range(m)]
    
    # Outer product: iterate over each k in the common dimension.
    for k in range(kA):
        col_start = indptrA[k]
        col_end   = indptrA[k+1]
        if col_start == col_end:
            continue

        row_start = indptrB[k]
        row_end   = indptrB[k+1]
        if row_start == row_end:
            continue

        # For each nonzero element A[i, k] in column k:
        for idx_a in range(col_start, col_end):
            i = indicesA[idx_a]
            a_val = dataA[idx_a]
            totalOPs += 1
            # For each nonzero element B[k, j] in row k:
            for idx_b in range(row_start, row_end):
                j = indicesB[idx_b]
                b_val = dataB[idx_b]
                totalOPs += 1
                C[i][j] += a_val * b_val
    print("outer product with CSCxCSR inputs number of operations: ", totalOPs+conversion_overhead, " (", totalOPs, "+", conversion_overhead, ")")
    return C


def csr_gustavson(csrA, csrB):
    dataA, indicesA, indptrA, shapeA, input_conversion_overheadA = csrA
    dataB, indicesB, indptrB, shapeB, input_conversion_overheadB = csrB
    m, kA = shapeA
    kB, n = shapeB
    conversion_overhead = input_conversion_overheadA + input_conversion_overheadB
    totalOPs = 0
    if kA != kB:
        raise ValueError("Inner dimensions do not match for multiplication.")
    
    C = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        row_acc = [0] * n
        # For every nonzero A[i,k] in row i of A:
        for a_idx in range(indptrA[i], indptrA[i+1]):
            k = indicesA[a_idx]
            a_val = dataA[a_idx]
            totalOPs += 1;
            # Get row k of B from its CSR structure.
            for b_idx in range(indptrB[k], indptrB[k+1]):
                j = indicesB[b_idx]
                b_val = dataB[b_idx]
                row_acc[j] += a_val * b_val
                totalOPs += 1;
        C[i] = row_acc
    print("gustavson algo with CSRxCSR input number of operations: ", totalOPs+conversion_overhead, " (", totalOPs, "+", conversion_overhead, ")")
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
    overhead = 0;
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    for row in matrix:
        for j, value in enumerate(row):
            if value != 0:
                data.append(value)
                indices.append(j)
                overhead += 1;
        # After processing a row, record the current length of data as the end pointer for that row.
        indptr.append(len(data))
    
    return data, indices, indptr, (m, n), overhead

def dense_to_csc(matrix):
    data = []
    indices = []
    indptr = [0]
    overhead = 0;
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    # Count nonzeros per column
    column_data = {j: [] for j in range(n)}
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value != 0:
                column_data[j].append((i, value))
                overhead += 1;
    
    # Build CSC arrays
    for j in range(n):
        for i, value in column_data[j]:
            data.append(value)
            indices.append(i)
            overhead += 1;
        indptr.append(len(data))
    
    return data, indices, indptr, (m, n), overhead

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
    cscC = dense_to_csc(C)
    csrD = dense_to_csr(D)
    cscD = dense_to_csc(D)

    C_CSRxCSC_inner = inner_product_csr_csc(csrC, cscD)
    C_CSCxCSR_outer = outer_product_csc_csr(cscC, csrD)
    C_csr_gustavson = csr_gustavson(csrC, csrD)
    
    
    same = (C_inner == C_outer == C_gustavson == C_CSRxCSC_inner == C_CSCxCSR_outer == C_csr_gustavson)
    print("All methods produce the same result:", same)
'''
    same2 = C_gustavson == C_csr_gustavson
    print("Gustavson CSR format matches naive format output: ", same2)
    same3 = C_CSRxCSC_inner == C_inner
    print("optimized inner matches naive inner output: ", same3)
'''

if __name__ == '__main__':
    main()
