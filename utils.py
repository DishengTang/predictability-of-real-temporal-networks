# %%
import numpy as np
from numpy import log2
import pandas as pd
import sympy as sp
from sympy import Symbol, nsolve
import sys

def load_data(path):
    return pd.read_excel(path, header=None, index_col=None).to_numpy()

def is_submatrix_elsewhere(submatrix, matrix):
    # use sliding_window_view to obtain sliding matrix
    if matrix.shape[0] >= submatrix.shape[0] and matrix.shape[1] >= submatrix.shape[1]:
        sliding_mat = np.lib.stride_tricks.sliding_window_view(matrix, window_shape=submatrix.shape)
        diff = np.abs(sliding_mat - submatrix[None, None, :, :]).sum(axis=-1).sum(axis=-1)
        diff[-1, -1] = 1 # this position is itself
        appeared_before = np.isin(0, diff)
    else:
        appeared_before = False
    return appeared_before

def split(matrix):
    flag = 1     #flag == 1 means there're still matrixs to split, 0 means none
    matrix_cell = []
    all_matrix_cell = []
    big_matrix_ind = []    #store the serial number of matrixs bigger than 1
    row,col = matrix.shape
    row_split,col_split = matrix.shape
    while row_split != col_split:
        matrix_cell.append(matrix[:min(row_split, col_split), :min(row_split, col_split)])
        if row_split < col_split:
            matrix = matrix[:,row_split:]
        else:
            matrix = matrix[col_split:,:]
        row_split,col_split = matrix.shape
    matrix_cell.append(matrix)
    while flag == 1:
        cnt = 0
        flag = 0
        for ind in range(len(matrix_cell)):
            if matrix_cell[ind].size > 1:
               cnt += matrix_cell[ind].size
               big_matrix_ind.append(ind)
               flag = 1
            elif [mat.size for mat in matrix_cell].count(1) <= 1:
               cnt += matrix_cell[ind]
            else:
               cnt += 1
#                matrix_cell{ind} = 1  #make sure subsequent split result is right
        if cnt != row * col:
            sys.exit('Sorry! The split result is wrong!!!')
        #emerge length 1 matrix
        loc = np.where(np.array([mat.size for mat in matrix_cell])==1)[0]
        if len(loc) > 1:
            matrix_cell[loc[0]] = np.array(len(loc)) # so that it has size method
            matrix_cell[loc[0] + 1:] = []
        
        all_matrix_cell.append(matrix_cell)
        if flag == 0:   #there is no need to split
            break
        #Split the minimum matrix bigger than 1
        temp = matrix_cell[:big_matrix_ind[-1]]
        if matrix_cell[-1].size == 1:
            temp.append(np.array(matrix_cell[-1] + matrix_cell[big_matrix_ind[-1]].size))
        else:
            temp.append(np.array(matrix_cell[big_matrix_ind[-1]].size))
        
        matrix_cell = temp
        big_matrix_ind = []
    return all_matrix_cell

def entropy_rate(p, uniq):
    if isinstance(p, (int, float, np.ndarray)):
        er = -(p * log2(p) + (1 - p) * log2(1 - p) - (1 - p) * log2(uniq - 1))
    elif isinstance(p, Symbol):
        er = -(p * sp.log(p, 2) + (1 - p) * sp.log(1 - p, 2) - (1 - p) * log2(uniq - 1))
    else:
        print(p)
        sys.exit('Incorrect type of p: {}'.format(type(p)))
    return er

def compute_square_predictability(square, cell_ind, cell_len):
    row, col = square.shape
    summ = 0
    for i in range(1, row):
        for j in range(1, col):
            for L in range(min(i, j)):
                matrix = square[:i, :j]
                submatrix = square[i-L:i,j-L:j]     # length L
                appeared_before = is_submatrix_elsewhere(submatrix, matrix)
                if not appeared_before:
                    summ += L ** 2
                    break
                elif L == min(i, j):                # max + 1 if max length still submatrix
                    summ += (L + 1) ** 2
        print('Row {}/{} of spliting cell {}/{} finished'.format(i, row, cell_ind + 1, cell_len))
    # entropy rate estimated by Lempel-Ziv algorithm
    H = square.size * log2(square.size) / summ
    uniq = len(np.unique(square))
    if uniq < 2:
        predictability = 1
    else:
        ps = np.arange(0.001,1,0.001)
        e = entropy_rate(ps, uniq)
        if H > e.max():
            predictability = 1 / uniq
        else:
            p = Symbol('p')
            f = H - entropy_rate(p, uniq)
            solution = nsolve(f, p, 0.5)
            predictability = complex(solution).real
    if uniq == 2 and predictability < 0.5:
        predictability = 1 - predictability
    return predictability

def compute_TTP(data):
    data = np.nan_to_num(data) # replace NaN as 0
    np.random.shuffle(data) # reorder the rows in the matrix
    uniq = len(np.unique(data))
    row,col = data.shape
    wavg_pred = 0  # weighted average predictability for certain split strategy
    matrix_cell = {}  # certain split strategy
    num_squares_list = [] # number of squares for each split strategy
    wavg_pred_list = [] # weighted average predictability for each split strategy
    origin_pred = [] # original predictability of the first split strategy
    if row != col:
        all_matrix_cell = split(data)
        for cell_ind in range(len(all_matrix_cell)):
            matrix_cell = all_matrix_cell[cell_ind]
            wavg_pred = 0
            for mat_ind in range(len(matrix_cell)):
                square = matrix_cell[mat_ind]
                row_norm,col_norm = square.shape
                if row_norm == 1:
                    pred = square / (row * col * uniq)
                elif cell_ind == 0:
                    pred = compute_square_predictability(square, cell_ind, len(all_matrix_cell))                        
                    origin_pred.append(pred)
                else:
                    if square == all_matrix_cell[0][mat_ind]: # find the first split predictability
                        pred = origin_pred[mat_ind] * square.size / (row * col)  #weighted average by area
                    else:
                        sys.exit('Sorry! There is some problem with the original split!!!')
                wavg_pred += pred * square.size / (row * col)  #weighted average by area
            if matrix_cell[-1].size == 1:
                num_squares = int(len(matrix_cell) - 1 + matrix_cell[-1])
            else:
                num_squares = len(matrix_cell)
            num_squares_list.append(num_squares)
            wavg_pred_list.append(wavg_pred)
        # use the linear correlation between weighted average
        # predictability and number of squares to infer actual
        # predicatbility
        fitting = np.polyfit(num_squares_list, wavg_pred_list, 1)
        predictability = fitting[0] + fitting[1]
    else: # no need to split matrix for square data
        predictability = compute_square_predictability(data, 1, 1)
    return predictability
        
