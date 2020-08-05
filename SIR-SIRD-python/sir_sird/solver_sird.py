from models.sir import SirModel
from models.sird import SirdModel

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
    parser.add_argument("--lr", default=1e-18, type=float)
    parser.add_argument("--epochs", default=1000, type=int)

    return parser.parse_args()


def train(model, df, population, lr, args):
    """ Train simple model 
    Args:
        model (nn.Module)
        df (pd.DataFrame): Data to train
        population (int): Population of country in csv file
    """

    # Load data
    r = df.recover
    r = np.array(r)
    d = np.array(df.death)
    i = np.array(df.confirmed) - r
    s = (-1) * (i + r - population)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.MSELoss(reduction="mean")

    n_s = len(s)
    s_prev = torch.tensor(s[0 : n_s - 1]).float()
    i_prev = torch.tensor(i[0 : n_s - 1]).float()
    r_prev = torch.tensor(r[0 : n_s - 1]).float()
    d_prev = torch.tensor(d[0 : n_s - 1]).float()

    s_next = torch.tensor(s[1:]).float()
    i_next = torch.tensor(i[1:]).float()
    r_next = torch.tensor(r[1:]).float()
    d_next = torch.tensor(d[1:]).float()

    for epoch in range(args.epochs):
        s_pred, i_pred, r_pred, d_pred = model(s_prev, i_prev, r_prev, d_prev, population)
        ls = loss_function(s_pred, s_next) 
        # ls = 0 
        li = loss_function(i_pred, i_next) 
        # li = 0
        lr = loss_function(r_pred, r_next) 
        ld = loss_function(d_pred, d_next)

        loss = ls + li + lr + ld
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
    model = SirdModel()

    df, population = load_df_and_population(args)

    train(model, df, population, args.lr, args)

    print(model)


if __name__ == "__main__":
    main()
