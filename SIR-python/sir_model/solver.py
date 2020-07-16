from models.sir import SirModel

import torch
from torch import nn
from torch import optim
import pandas as pd
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Training script for Sir Model")
    parser.add_argument("csv_path")
    parser.add_argument("population", type=int)

    parser.add_argument("--start_day", default=30, type=int)
    parser.add_argument("--end_day", default=60, type=int)
    parser.add_argument("--lr", default=1e-14, type=float)
    parser.add_argument("--epochs", default=100, type=int)

    return parser.parse_args()


def train(model, df, population, lr, args):
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

    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.MSELoss(reduction="mean")

    n_s = len(s)
    s_prev = torch.tensor(s[0 : n_s - 1]).float()
    i_prev = torch.tensor(i[0 : n_s - 1]).float()
    r_prev = torch.tensor(r[0 : n_s - 1]).float()

    s_next = torch.tensor(s[1:]).float()
    i_next = torch.tensor(i[1:]).float()
    r_next = torch.tensor(r[1:]).float()

    max_s = np.max(s)
    max_r = np.max(r)
    max_i = np.max(i)

    for epoch in range(args.epochs):
        s_pred, i_pred, r_pred = model(s_prev, i_prev, r_prev, population)
        ls = loss_function(s_pred, s_next) 
        # ls = 0 
        li = loss_function(i_pred, i_next) 
        # li = 0
        lr = loss_function(r_pred, r_next) 

        loss = ls + li + lr
        loss_int = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (args.epochs // 10) == 0:
            print(f"Epoch {epoch}: {loss_int}")


def load_df_and_population(args):
    df = pd.read_csv(args.csv_path)
    population = int(args.population)

    df = df[args.start_day : args.end_day + 1]

    return df, population


def main():
    args = parse_args()
    model = SirModel()

    df, population = load_df_and_population(args)

    train(model, df, population, args.lr, args)

    print(model)


if __name__ == "__main__":
    main()
