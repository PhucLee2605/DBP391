import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def confusionMatrix(y_true, y_pre):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pre)

    recall = cm[1,1] / (cm[1,1] + cm[1,0]) * 100
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) * 100

    f1_score = 2 * (recall * precision) / (recall + precision)

    print(f"recall : {recall}\nprecision : {precision}\nf1 score : {f1_score}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()