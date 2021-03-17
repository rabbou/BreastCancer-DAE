# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define global variables
ROOT    = "C:/Users/Mario/Google Drive/Documents/University of Chicago/Master/Classes/1st Year/TTIC 31220/BC_TTIC_Project/BreastCancer-DAE/" 
RESULTS = ROOT + "baseline_results/"

def GraphResults(size = 32):
    df40  = pd.read_csv(RESULTS + f"{size}_40_acc.csv", sep = ",", header = None).values
    df100 = pd.read_csv(RESULTS + f"{size}_100_acc.csv", sep = ",", header = None).values
    df200 = pd.read_csv(RESULTS + f"{size}_200_acc.csv", sep = ",", header = None).values
    df400 = pd.read_csv(RESULTS + f"{size}_400_acc.csv", sep = ",", header = None).values

    train_acc = pd.DataFrame()
    train_acc["40X"]  = df40[0, :]
    train_acc["100X"] = df100[0, :]
    train_acc["200X"] = df200[0, :]
    train_acc["400X"] = df400[0, :]

    valid_acc = pd.DataFrame()
    valid_acc["40X"]  = df40[1, :]
    valid_acc["100X"] = df100[1, :]
    valid_acc["200X"] = df200[1, :]
    valid_acc["400X"] = df400[1, :]

    train_f1 = pd.DataFrame()
    train_f1["40X"]  = df40[2, :]
    train_f1["100X"] = df100[2, :]
    train_f1["200X"] = df200[2, :]
    train_f1["400X"] = df400[2, :]

    valid_f1 = pd.DataFrame()
    valid_f1["40X"]  = df40[3, :]
    valid_f1["100X"] = df100[3, :]
    valid_f1["200X"] = df200[3, :]
    valid_f1["400X"] = df400[3, :]

    fig, axs = plt.subplots(2, 2, figsize = (8, 8), constrained_layout = True)
    axs[0, 0].plot(train_acc.index, train_acc)
    axs[0, 0].legend(train_acc.columns, loc = "lower right")
    axs[0, 0].set_ylabel("Accuracy")
    axs[0, 0].set_title("Train")
    axs[0, 0].set_xticks([])
    axs[0, 0].set_ylim(60, 100)

    axs[0, 1].plot(valid_acc.index, valid_acc)
    axs[0, 1].legend(valid_acc.columns, loc = "lower right")
    axs[0, 1].set_title("Validation")
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_ylim(60, 100)

    axs[1, 0].plot(train_f1.index, train_f1)
    axs[1, 0].legend(train_f1.columns, loc = "lower right")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("F1")
    axs[1, 0].set_ylim(0.5, 1)

    axs[1, 1].plot(valid_f1.index, valid_f1)
    axs[1, 1].legend(valid_f1.columns, loc = "lower right")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_yticks([])
    axs[1, 1].set_ylim(0.5, 1)
    plt.savefig(f"BaselineResults_{size}.png")
    return

GraphResults(size = 32)
GraphResults(size = 64)