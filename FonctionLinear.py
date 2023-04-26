import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import sys

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

def preprocessing(datasets_file):
    try:
        datasets = pd.read_csv(datasets_file, encoding='utf8', engine='python')
        x = datasets.iloc[0:len(datasets), 0]
        x = x.values.reshape(x.shape[0], 1)
        y = datasets.iloc[0:len(datasets), 1]
        y = y.values.reshape(y.shape[0], 1)
        return x, y
    except:
        raise BaseException(f"Datasets File { datasets_file } should trigger an error")
    
def linearFunction(theta0, theta1, km):
    return theta0 + (theta1 * km)

def checkPositive(value):
    floatValue = float(value)
    if floatValue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return floatValue

def getThetaPred():
    theta0 = theta1 = np.zeros((1, 1))
    thetaPredFile = Path("theta.csv")
    if thetaPredFile.is_file():
        try:
            thetaPred = pd.read_csv('theta.csv')
            theta0 = thetaPred.iloc[0:1, 0].values.reshape(1, 1)
            theta1 = thetaPred.iloc[0:1, 1].values.reshape(1, 1)
            if ((theta0[0]) is None) or ((theta1[0]) is None):
                theta0 = theta1 = np.zeros((1, 1))
        except:
            raise BaseException(f"File theta.csv should trigger an error")
    return theta0, theta1

def estimatePrice(datasets_file, mileage):
    x, y = preprocessing(datasets_file)
    mileage = (mileage - min(x)) / (max(x) - min(x))
    theta0, theta1 = getThetaPred()
    estimatePrice = np.around(linearFunction(theta0, theta1, mileage), 2)
    return estimatePrice, theta0, theta1, x, y

def graphCompare(x, y, X_test, Y_test):
    plt.scatter(x, y, s=10, c='b', marker="s", label='Dataset')
    plt.scatter(X_test, Y_test, s=10, c='r', marker="o", label='Predicted')
    plt.legend(loc='upper right')
    plt.title('Linear Regression')
    plt.xlabel('mileage')
    plt.ylabel('price')
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LinearRegression @Paris 42 School - Made by @bhamdi')
    parser.add_argument("input", type=str, help="The file containing datasets")
    parser.add_argument("m", type=checkPositive, help="Mileage")
    parser.add_argument("-g", "--graph", action="store_true", help="Show model")
    args = parser.parse_args()
    try:
        estimatePrice, theta0, theta1, x, y = estimatePrice(args.input, args.m)
        print(f"[theta0: {Color.BLUE}{round(theta0[0][0], 2)}{Color.END}] [theta1: {Color.BLUE}{round(theta1[0][0], 2)}{Color.END}]")
        print(f"[Mileage: {Color.GREEN}{args.m}{Color.END}] [Estimated Price: {Color.GREEN}{estimatePrice[0][0]}{Color.END}] ")
        if (args.graph):
            graphCompare(x, y, args.m, estimatePrice)
    except (Exception, BaseException) as e:
        print(f"{Color.WARNING} {e} {Color.END}")
        sys.exit(1)
