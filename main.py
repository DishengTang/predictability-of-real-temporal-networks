# Script for calculating the predicatbility (TTP/NTTP) of temporal networks
# Tang, D., Du, W., Shekhtman, L., Wang, Y., Havlin, S., Cao, X., & Yan, G. (2020). 
# Predictability of real temporal networks. National Science Review, 7(5), 929-937.

import numpy as np
import argparse
from utils import compute_predictability
import sys
import time
import os
from data_preprocess import DataProcessor

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate (normalized) topological-temporal predictability')
    parser.add_argument('-dp', '--DataPath', required=True, help='Path to the data file, e.g. ./raw_data/forum.txt')
    parser.add_argument('-l','--DataColumn', nargs='+', required=True, help='Specify the format of the data')
    parser.add_argument('-fn', '--FileName', required=True, help='File name to save the result, e.g. CF')
    parser.add_argument('-n', '--Normalize', default=0, type = int, choices=[0, 1], help='Calculated NTTP or TTP, 1 for NTTP')
    parser.add_argument('-nn', '--NumNorm', default=100, type = int, help='Number of TTP baseline realizations')
    parser.add_argument('-conv', '--UseConv', default=1, type = int, choices=[0, 1], help='Whether to use Convolution for submatrix matching, 1 for use')
    parser.add_argument('-dr', '--Directed', default=1, type = int, choices=[0, 1], help='Whether the network is directed, 1 for directed')
    parser.add_argument('-fre', '--FreNode', default=0, type = int, choices=[0, 1], help='Whether to remove nodes with occurences under threshold, 1 for use')
    parser.add_argument('-th', '--Threshold', default=10, type = int, help='Threshold of minimum occurences for nodes')
    parser.add_argument('-cc', '--ConCom', default=0, type = int, choices=[0, 1], help='Whether to use connected component to filter nodes, 1 for use')
    parser.add_argument('-ft', '--Filter', default=1, type = int, choices=[0, 1], help='Whether to filter active links, 1 for use')
    args = parser.parse_args()
    start_time = time.time()
    pred_name = 'NTTP' if args.Normalize else 'TTP'
    data_process = DataProcessor(args.DataPath, args.DataColumn, is_directed=args.Directed, frequent_node=args.FreNode,
            th=args.Threshold, connected_component=args.ConCom, filter=args.Filter)
    data_process.load_data()
    data_process.construct_matrix()
    data_process.filter_links()
    predictability = compute_predictability(data_process.M_tilde, args.Normalize, args.NumNorm, args.UseConv)
    f = open(args.FileName, "w+")
    f.write(str(predictability))
    f.close()
    print('{} for {} is :{}'.format(pred_name, args.DataPath, predictability))
    print('It takes {} minites to execute this project'.format(round((time.time() - start_time) / 60, 1)))