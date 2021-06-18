# script for calculating the predicatbility (TTP) of temporal networks
# Tang, D., Du, W., Shekhtman, L., Wang, Y., Havlin, S., Cao, X., & Yan, G. (2020). 
# Predictability of real temporal networks. National Science Review, 7(5), 929-937.

import numpy as np
import argparse
from utils import load_data, compute_TTP
import sys
import time
import os
from data_preprocess import DataProcessor

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Calculate topological-temporal predictability (TTP)')
    # parser.add_argument('-dp', '--DataPath', required=True, help='Path to the data file (only excel file is accepted), e.g. ./data/CF.xlsx')
    # parser.add_argument('-fn', '--FileName', required=False, help='File name to save the result, e.g. ./TTP/CF.txt')
    # parser.add_argument('-conv', '--UseConv', default=0, type = int, choices=[0, 1], help='Whether to use Convolution for submatrix matching')
    # args = parser.parse_args()
    # start_time = time.time()
    # for file in os.listdir(args.DataPath):
    #     if file.endswith(".xlsx"):
    #         print(file)
    #         data = load_data(os.path.join(args.DataPath, file))
    #         TTP = compute_TTP(data, args.UseConv)
    #         f = open('./TTP_calculated/'+file.replace('.xlsx', '.txt'), "w+")
    #         f.write(str(TTP))
    #         f.close()
    #         print('TTP for {} is :{}'.format(file.replace('.xlsx', ''), TTP))
    # print('It takes {} minites to execute this project'.format(round((time.time() - start_time) / 60, 1)))

    parser = argparse.ArgumentParser(description='Calculate topological-temporal predictability (TTP)')
    parser.add_argument('-dp', '--DataPath', required=True, help='Path to the data file (only excel file is accepted), e.g. ./data/CF.xlsx')
    parser.add_argument('-l','--DataColumn', nargs='+', required=True, help='Specify the format of the data')
    parser.add_argument('-fn', '--FileName', required=True, help='File name to save the result, e.g. ./TTP/CF.txt')
    parser.add_argument('-conv', '--UseConv', default=0, type = int, choices=[0, 1], help='Whether to use Convolution for submatrix matching, 1 means to use it')
    parser.add_argument('-dr', '--Directed', default=1, type = int, choices=[0, 1], help='Whether the network is directed')
    parser.add_argument('-fre', '--FreNode', default=0, type = int, choices=[0, 1], help='Whether to remove nodes with occurences under threshold, 1 means to use it')
    parser.add_argument('-th', '--Threshold', default=10, type = int, help='Threshold of minimum occurences for nodes')
    parser.add_argument('-cc', '--ConCom', default=0, type = int, choices=[0, 1], help='Whether to use connected component to filter nodes, 1 means to use it')
    parser.add_argument('-ft', '--Filter', default=1, type = int, choices=[0, 1], help='Whether to filter active links, 1 means to use it')
    args = parser.parse_args()
    start_time = time.time()
    # Data format
    # data_col = ['source', 'target', 'time']
    # data_col = ['source', 'target', 'weight', 'time']
    # path = args.DataPath
    data_process = DataProcessor(args.DataPath, args.DataColumn, is_directed=args.Directed, frequent_node=args.FreNode,
            th=args.Threshold, connected_component=args.ConCom, filter=args.Filter)
    data_process.load_data()
    data_process.construct_matrix()
    data_process.filter_links()
    TTP = compute_TTP(data_process.M_tilde, args.UseConv)
    f = open(args.FileName, "w+")
    f.write(str(TTP))
    f.close()
    print('TTP for {} is :{}'.format(args.DataPath, TTP))
    print('It takes {} minites to execute this project'.format(round((time.time() - start_time) / 60, 1)))
    # for file in os.listdir(args.DataPath):
    #     print(file)
    #     path = os.path.join(args.DataPath, file)
    #     path = args.DataPath
    #     data_process = DataProcessor(path, data_col, is_directed=args.Directed, frequent_node=args.FreNode,
    #             th=args.Threshold, connected_component=args.ConCom, filter=args.Filter)
    #     data_process.load_data()
    #     data_process.construct_matrix()
    #     data_process.filter_links()
    #     TTP = compute_TTP(data_process.M_tilde, args.UseConv)
    #     f = open('./TTP_calculated/'+file.replace('.xlsx', '.txt'), "w+")
    #     f.write(str(TTP))
    #     f.close()
    #     print('TTP for {} is :{}'.format(file.replace('.xlsx', ''), TTP))
    # print('It takes {} minites to execute this project'.format(round((time.time() - start_time) / 60, 1)))