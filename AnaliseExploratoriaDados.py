import pandas as pd
import numpy as np 
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def main():
    df = pd.read_csv('C:\\Users\\Rafael\\Downloads\\creditcard.csv')
    
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())
    
    print('Fraude')
    print(df.Time[df.Class == 1].describe())
    print()
    print('Normal')
    print(df.Time[df.Class == 0].describe())
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (12, 4))
    ax1.hist([n for n in df.Time[df.Class == 1]], bins = 50)
    ax1.set_title('Fraude')
    ax2.hist([n for n in df.Time[df.Class == 0]], bins = 50)
    ax2.set_title('Normal')
    plt.xlabel('Tempo (em segundos)')
    plt.ylabel('Número de transações')
    plt.show()
    
    print('Fraude')
    print(df.Amount[df.Class == 1].describe())
    print()
    print('Normal')
    print(df.Amount[df.Class == 0].describe())
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (12, 4))
    ax1.hist([n for n in df.Amount[df.Class == 1]], bins = 30)
    ax1.set_title('Fraude')
    ax2.hist([n for n in df.Amount[df.Class == 0]], bins = 30)
    ax2.set_title('Normal')
    plt.xlabel('Montante (€)')
    plt.ylabel('Número de transações')
    plt.yscale('log')
    plt.show()
    
    df['Amount_max_fraud'] = 1
    df.loc[df.Amount <= 2125.87, 'Amount_max_fraud'] = 0
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (12, 6))
    ax1.scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1])
    ax1.set_title('Fraude')
    ax2.scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0])
    ax2.set_title('Normal')
    plt.xlabel('Tempo (em segundos)')
    plt.ylabel('Montante (€)')
    plt.show()
    
    for feature in df.ix[:,1:29].columns:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.distplot(df[feature][df.Class == 1], bins = 50, label = 'Fraude')
        sns.distplot(df[feature][df.Class == 0], bins = 50, label = 'Normal')
        ax.set_xlabel('')
        ax.set_title('Histograma do atributo (feature): ' + str(feature))
        plt.legend(loc = 'best')
        plt.show()
    
if __name__ == '__main__':
    main()