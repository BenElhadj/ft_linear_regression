import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys

old_settings = np.seterr(all='ignore')
class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

class Val:
    nIteration = 10000
    learning_rate = 0.01

def preprocessing(datasets_file):
    try:
        datasets = pd.read_csv(datasets_file, encoding='utf8', engine='python')
        x = datasets.iloc[0:len(datasets), 0]
        x = x.values.reshape(x.shape[0], 1)
        xNormalized = (x - min(x)) / (max(x) - min(x))   
        y = datasets.iloc[0:len(datasets), 1]
        y = y.values.reshape(y.shape[0], 1)
    except:
        raise BaseException(f"Datasets File { datasets_file } should trigger an error") 
    if pd.isna(x).any() or pd.isna(y).any():
        raise BaseException(f"Datasets File { datasets_file } should trigger an error")

    X = np.hstack((xNormalized, np.ones(xNormalized.shape)))
    return X, x, y

def checkPositive(value):
    floatValue = float(value)
    if floatValue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return floatValue


def coefCorrelation(x, y):
    r = ((x - x.mean()) * (y - y.mean())).sum()
    v = np.sqrt(((x - x.mean())**2).sum()) * np.sqrt(((y - y.mean())**2).sum())
    result = r/v if v !=0 else 0
    return result

def coefDetermination(y, pred):
    r = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    result = 1 - r/v if v !=0 else 0
    return result

def costFunction(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

def model(X, theta):
    return X.dot(theta)

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

def gradientDescent(X, y, learning_rate, nIteration):
    theta = np.zeros((2, 1))
    costHistory = np.zeros(nIteration)
    for i in range(0, nIteration):
        theta = theta - learning_rate * grad(X, y, theta)
        costHistory[i] = costFunction(X, y, theta)
    return theta, costHistory

def saveTheta(thetaPred):
    df = pd.DataFrame(data={"theta0": thetaPred[1], "theta1": thetaPred[0]})
    df.to_csv("theta.csv", sep=',', index=False)

def graph(pred, x, y, costHistory):
    plt.figure(1)
    plt.scatter(x, y, marker='x')
    plt.title('Linear Regression')
    plt.xlabel('mileage')
    plt.ylabel('price')
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.plot(x, pred, c='r')
    plt.figure(2)
    plt.plot(range(Val.nIteration), costHistory)
    plt.title('Cost History')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()

def predictions(datasetsFile, learningRate):
    X, x, y = preprocessing(datasetsFile)  
    thetaPred, costHistory = gradientDescent(X, y, learningRate, Val.nIteration)
    pred = model(X, thetaPred)
    coef = coefDetermination(y, pred)
    coef_corr = coefCorrelation(x, y)
    saveTheta(thetaPred)
    return thetaPred, coef, coef_corr, pred, x, y, costHistory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LinearRegression @Paris 42 School - Made by @bhamdi')
    parser.add_argument("input", type=str, help="The file containing datasets")
    parser.add_argument("-r", "--rate", type=checkPositive, help="custom learning rate")
    parser.add_argument("-g", "--graph", action="store_true", help="Show model")
    args = parser.parse_args()
    try:
        if (args.rate is not None):
            thetaPred, coef, coef_corr, pred, x, y, costHistory = predictions(args.input, args.rate)
        else:
            thetaPred, coef, coef_corr, pred, x, y, costHistory = predictions(args.input, Val.learning_rate)
        print(f"[{Color.BLUE}Model Trained{Color.END}]")
        print(f"[theta0: {Color.GREEN}{round(thetaPred[1][0], 2)}{Color.END}] [theta1: {Color.GREEN}{round(thetaPred[0][0],2)}{Color.END}]")
        print(f"[Coefficient de Determination: {Color.BLUE}{round(coef, 3)}{Color.END}]")
        print(f"[Coefficient de Correlation: {Color.BLUE}{round(coef_corr, 3)}{Color.END}]")
        if (args.graph):
            graph(pred, x, y, costHistory)
    except (Exception, BaseException) as e:
        print(f"{Color.WARNING} {e} {Color.END}")
        sys.exit(1)