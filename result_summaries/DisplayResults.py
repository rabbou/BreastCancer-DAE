# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define global variables
ROOT    = "C:/Users/Mario/Google Drive/Documents/University of Chicago/Master/Classes/1st Year/TTIC 31220/BC_TTIC_Project/BreastCancer-DAE/" 

def GraphResults(size, baseline):
    if baseline:
        results = ROOT + "baseline_results/"
    else:
        results = ROOT + "results/"
    df40  = pd.read_csv(results + f"{size}_40_acc.csv", sep = ",", header = None).values
    df100 = pd.read_csv(results + f"{size}_100_acc.csv", sep = ",", header = None).values
    df200 = pd.read_csv(results + f"{size}_200_acc.csv", sep = ",", header = None).values
    df400 = pd.read_csv(results + f"{size}_400_acc.csv", sep = ",", header = None).values

    train_acc = pd.DataFrame()
    train_acc["40X"]  = df40[0, :]
    train_acc["100X"] = df100[0, :]
    train_acc["200X"] = df200[0, :]
    train_acc["400X"] = df400[0, :]

    test_acc = pd.DataFrame()
    test_acc["40X"]  = df40[1, :]
    test_acc["100X"] = df100[1, :]
    test_acc["200X"] = df200[1, :]
    test_acc["400X"] = df400[1, :]

    train_f1 = pd.DataFrame()
    train_f1["40X"]  = df40[2, :]
    train_f1["100X"] = df100[2, :]
    train_f1["200X"] = df200[2, :]
    train_f1["400X"] = df400[2, :]

    test_f1 = pd.DataFrame()
    test_f1["40X"]  = df40[3, :]
    test_f1["100X"] = df100[3, :]
    test_f1["200X"] = df200[3, :]
    test_f1["400X"] = df400[3, :]

    fig, axs = plt.subplots(2, 2, figsize = (8, 8), constrained_layout = True)
    axs[0, 0].plot(train_acc.index, train_acc)
    axs[0, 0].legend(train_acc.columns, loc = "lower right")
    axs[0, 0].set_ylabel("Accuracy")
    axs[0, 0].set_title("Training")
    axs[0, 0].set_xticks([])
    axs[0, 0].set_ylim(60, 100)

    axs[0, 1].plot(test_acc.index, test_acc)
    axs[0, 1].legend(test_acc.columns, loc = "lower right")
    axs[0, 1].set_title("Testing")
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_ylim(60, 100)

    axs[1, 0].plot(train_f1.index, train_f1)
    axs[1, 0].legend(train_f1.columns, loc = "lower right")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("F1-Score")
    axs[1, 0].set_ylim(0.5, 1)

    axs[1, 1].plot(test_f1.index, test_f1)
    axs[1, 1].legend(test_f1.columns, loc = "lower right")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_yticks([])
    axs[1, 1].set_ylim(0.5, 1)
    
    if baseline:
        plt.savefig(f"BaselineResults_{size}.png")
    else:
        plt.savefig(f"Results_{size}.png")
    return

def GetBestResults(size, baseline):
    if baseline:
        results = ROOT + "baseline_results/"
    else:
        results = ROOT + "results/"
    df40  = pd.read_csv(results + f"{size}_40_acc.csv", sep = ",", header = None).values
    df100 = pd.read_csv(results + f"{size}_100_acc.csv", sep = ",", header = None).values
    df200 = pd.read_csv(results + f"{size}_200_acc.csv", sep = ",", header = None).values
    df400 = pd.read_csv(results + f"{size}_400_acc.csv", sep = ",", header = None).values

    train_acc = pd.DataFrame()
    train_acc["40X"]  = df40[0, :]
    train_acc["100X"] = df100[0, :]
    train_acc["200X"] = df200[0, :]
    train_acc["400X"] = df400[0, :]

    test_acc = pd.DataFrame()
    test_acc["40X"]  = df40[1, :]
    test_acc["100X"] = df100[1, :]
    test_acc["200X"] = df200[1, :]
    test_acc["400X"] = df400[1, :]

    train_f1 = pd.DataFrame()
    train_f1["40X"]  = df40[2, :]
    train_f1["100X"] = df100[2, :]
    train_f1["200X"] = df200[2, :]
    train_f1["400X"] = df400[2, :]

    test_f1 = pd.DataFrame()
    test_f1["40X"]  = df40[3, :]
    test_f1["100X"] = df100[3, :]
    test_f1["200X"] = df200[3, :]
    test_f1["400X"] = df400[3, :]

    BestResults = pd.DataFrame()
    BestResults["TrainAcc"] = train_acc.max()
    BestResults["TestAcc"]  = test_acc.max()
    BestResults["TrainF1"]  = train_f1.max()
    BestResults["TestF1"]   = test_f1.max()

    if baseline:
        BestResults.to_csv(f"BestBaselineResults_{size}.csv")
    else:
        BestResults.to_csv(f"BestResults_{size}.csv")
    return

GraphResults(32, True)
GraphResults(64, True)
GraphResults(32, False)
GraphResults(64, False)

GetBestResults(32, True)
GetBestResults(64, True)
GetBestResults(32, False)
GetBestResults(64, False)