# script for calculating the predicatbility (TTP) of temporal networks
# Tang, D., Du, W., Shekhtman, L., Wang, Y., Havlin, S., Cao, X., & Yan, G. (2020). 
# Predictability of real temporal networks. National Science Review, 7(5), 929-937.

import numpy as np
import argparse
from utils import load_data, compute_TTP
import sys
import time
import os

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Calculate topological-temporal predictability (TTP)')
    # parser.add_argument('-dp', '--DataPath', required=True, help='Path to the data file (only excel file is accepted), e.g. ./data/CF.xlsx')
    # parser.add_argument('-fn', '--FileName', help='File name to save the result, e.g. ./TTP/CF.txt')
    # args = parser.parse_args()
    # start_time = time.time()
    # data = load_data(args.DataPath)
    # TTP = compute_TTP(data)
    # print('TTP is :{}'.format(TTP))
    # # f = open(args.FileName, 'a')
    # # f.writelines(TTP)
    # # f.close()
    # print('It takes {} minites to execute this project'.format(round((time.time() - start_time) / 60, 1)))


    parser = argparse.ArgumentParser(description='Calculate topological-temporal predictability (TTP)')
    parser.add_argument('-dp', '--DataPath', required=True, help='Path to the data file (only excel file is accepted), e.g. ./data/CF.xlsx')
    parser.add_argument('-fn', '--FileName', help='File name to save the result, e.g. ./TTP/CF.txt')
    args = parser.parse_args()
    start_time = time.time()
    
    for file in os.listdir(args.DataPath):
        if file.endswith(".xlsx"):
            print(file)
            data = load_data(os.path.join(args.DataPath, file))
            TTP = compute_TTP(data)
            f = open('./TTP/'+file.replace('.xlsx', '.txt'), "w+")
            f.write(str(TTP))
            f.close()
            print('TTP for {} is :{}'.format(file.replace('.xlsx', ''), TTP))
    print('It takes {} minites to execute this project'.format(round((time.time() - start_time) / 60, 1)))
