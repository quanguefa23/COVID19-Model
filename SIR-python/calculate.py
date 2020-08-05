import pandas as pd
from matplotlib import pyplot as plt
import argparse

import numpy as np

import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")


    return parser.parse_args()

def convert_to_vec(df, keys):
    vs = []
    for k in keys:
        vs.append(np.array(df[k]))

    vs = np.stack(vs)
    return vs

def main():
    args = parse_args()
    df = pd.read_csv(args.data_path)

    # Convert to vector
    l = ["s", "i", "r"]
    y_hat = convert_to_vec(df, l)
    y = convert_to_vec(df, [c + "_gt" for c in l])

    mae = 1 / len(y) * np.sum(np.abs(y - y_hat))
    nmae = 1 / len(y) * np.sum(np.abs(y - y_hat) / y)

    print(mae)
    print(nmae)
    





main()


