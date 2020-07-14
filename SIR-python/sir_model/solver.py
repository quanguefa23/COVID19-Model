from models.sir import SirModel

from torch import nn
from torch import optim
import pandas as pd
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Training script for Sir Model")
    parser.add_argument("csv_path")
    parser.add_argument("population")

    parser.add_argument("--start_day", default=30, type=int)
    parser.add_argument("--end_day", default=60, type=int)

    return parser.parse_args()


def train(model, df, population):
    """ Train simple model 
    Args:
        model (nn.Module)
        df (pd.DataFrame): Data to train
        population (int): Population of country in csv file
    """

    # Load data
    r = df.death + df.recover
    r = np.array(r)
    i = np.array(df.confirmed) - r
    s = (-1) * (i + r - population)

    optimizer = optim.SGD(model.paramters(), lr=1e-2)
    loss_function = nn.MSELoss()

    n_s = len(s)
    s_prev = s[0 : n_s - 1]
    i_prev = i[0 : n_s - 1]
    r_prev = r[0 : n_s - 1]

    s_next = s[1:]
    i_next = i[1:]
    r_next = r[1:]

    for epoch in range(100):
        s_pred, i_pred, r_pred = model(s_prev, i_prev, r_prev, population)
        ls = loss_function(s_pred, s_next)
        li = loss_function(i_pred, i_next)
        lr = loss_function(r_pred, r_next)

        loss = ls + li + lr
        loss_int = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {loss_int}")


def load_df_and_population(args):
    df = pd.read_csv(args.csv_path)
    population = int(args.population)

    df = df[args.start_day : args : end_day + 1]

    return df, population


def main():
    args = parse_args()
    model = SirModel()

    df, population = load_df_and_population(args)

    train(model, df, population)
    print(model)


if __name__ == "__main__":
    main()
