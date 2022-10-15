import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist



'''
    ####################################################################
    PLOT FOR VISUALIZATION 
    ####################################################################
'''

def singlePlot(df, row):
    Y = df.columns.to_list()
    X = df[:row].values.flatten().tolist()
    plt.figure(figsize=(12, 10))

    # Seaborn

    sns.lineplot(x=Y, y=X)

    # Setting Ticks

    plt.tick_params(axis='x', labelsize=15, rotation=90)
    plt.tight_layout()

    # Display

    plt.show()


def multiPlot(df, num, Xlabel, Ylabel, fig):
    muldf = pd.DataFrame()

    for i in range(num):
        muldf[i] = df.iloc[i, :].values

    plt.figure(figsize=fig)

    muldf.plot(subplots=True, xlabel=Xlabel, ylabel=Ylabel, legend=False, figsize=(40, 40))
    plt.show()


def plotKDistance(X, Y, idx, k):
    data = [[i, j] for i, j in zip(X, Y)]

    # Find K-distance
    dist_matrix = cdist(data, data)
    c = np.copy(dist_matrix[idx])
    c.sort()
    kdist = c[k]

    ###PLOT
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X, Y, s=50)

    # points inside circle
    points = np.take(data, [np.where(dist_matrix[idx] == i) for i in c[:k + 1]], axis=0)
    points = np.squeeze(points)
    ax.scatter(points[:, 0], points[:, 1], s=50, color='r')
    # Circle
    cir = plt.Circle(data[idx], kdist, color='black', fill=False)
    ax.set_aspect('equal', adjustable='datalim')
    ax.add_patch(cir)
    ax.set_title(f'K-distance={kdist:.2f} with k={k}', fontsize=22)
    plt.show()


'''
    ####################################################################
    PLOT FOR EVALUATION 
    ####################################################################
'''
def confusionMatrix(y_true, y_pre):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pre)
    print(cm)
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) * 100
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) * 100

    f1_score = 2 * (recall * precision) / (recall + precision)

    print(f"recall : {recall}\nprecision : {precision}\nf1 score : {f1_score}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()