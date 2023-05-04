import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data loading
def load_data():
    # Load data
    data_x = pd.DataFrame(pd.read_csv('./Data/Data_X.csv'))
    data_y = pd.DataFrame(pd.read_csv('./Data/Data_Y.csv'))
    data_new_x = pd.DataFrame(pd.read_csv('./Data/DataNew_X.csv'))
    return data_x, data_y, data_new_x

# 2. Checking the data
def check_data(data_x, data_y, data_new_x):
    print(f'\nDataX head: \n{data_x.head()}')
    print(f'\nDataX shape: \n{data_x.shape}')
    print(f'\nDataX types: \n{data_x.dtypes}')
    print(f'\nDataX missing values: \n{data_x.isnull().sum()}')
    print(f'\nDataX description: \n{data_x.describe()}')
    print(f'\nDataX number of rows with missing values: \n{data_x.isnull().any(axis=1).sum()}')

    print(f'\nDataNew_X head: \n{data_new_x.head()}')
    print(f'\nDataNew_X shape: \n{data_new_x.shape}')
    print(f'\nDataNew_X types: \n{data_new_x.dtypes}')
    print(f'\nDataNew_X description: \n{data_new_x.describe()}')
    print(f'\nDataNew_X missing values: \n{data_new_x.isnull().sum()}')
    print(f'\nDataNew_X number of rows with missing values: \n{data_new_x.isnull().any(axis=1).sum()}')

    print(f'\nDataY head: \n{data_y.head()}')
    print(f'\nDataY shape: \n{data_y.shape}')
    print(f'\nDataY types: \n{data_y.dtypes}')
    print(f'\nDataY description: \n{data_y.describe()}')
    print(f'\nDataY missing values: \n{data_y.isnull().sum()}')
    print(f'\nDataY number of rows with missing values: \n{data_y.isnull().any(axis=1).sum()}')


# 3. Data preparation
# 3.1. Removing the missing values
def remove_missing_values(data_x, data_y, data_new_x):
    new_data_x = data_x.dropna()
    new_data_y = data_y.dropna()
    new_data_new_x = data_new_x.dropna()
    return new_data_x, new_data_y, new_data_new_x

# 4. Explorative Data Analysis (EDA)
# 4.1. Overview of the data
def overview_data(dataframe):
    # Plotting the Histogram of the dataframe (remove the ids columns)
    dataframe = dataframe.drop('ID', axis=1)
    dataframe = dataframe.drop('DAY_ID', axis=1)
    dataframe = dataframe.drop('COUNTRY', axis=1)
    dataframe.hist(grid=False)
    plt.gcf().set_size_inches(20, 20)
    plt.title('Histogram of the dataframe')
    plt.show()

    # Plotting the Box plots of the dataframe
    dataframe.plot(kind='box', subplots=True, layout=(8,5), sharex=False, sharey=False)
    plt.title('Box plots of the dataframe')
    plt.gcf().set_size_inches(20, 20)
    plt.show()

    # Plotting the Scatter plots of the dataframe
    pd.plotting.scatter_matrix(dataframe, diagonal='hist')
    for ax in plt.gcf().axes:
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.xaxis.label.set_ha('right')
        ax.yaxis.label.set_ha('center')
        ax.yaxis.label.set_position((-5, 0.1))
        ax.yaxis.label.set_horizontalalignment('right')
        ax.xaxis.label.set_verticalalignment('top')
    plt.gcf().set_size_inches(20, 20)
    plt.suptitle('Scatter plots of the dataframe', fontsize=16)
    plt.gcf().set_size_inches(20, 20)
    plt.show()

    # Plotting the Correlation matrix of the dataframe`
    fig = plt.figure()
    fig.set_size_inches(20, 20)
    ax = fig.add_subplot(111)
    cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    ax.tick_params(axis='both', which='major', labelsize=10, pad=10)
    plt.title('Correlation matrix of the dataframe')
    plt.show()

    # Plotting the Heatmap for the Correlation matrix of the dataframe
    sns.heatmap(dataframe.corr(), annot=True, cmap='coolwarm', annot_kws={'size': 6}, fmt='.3f')
    plt.title('Heatmap for the Correlation matrix of the dataframe')
    #plt.gcf().set_size_inches(30, 30)
    plt.show()

def main():
    # 1. Load data
    data_x, data_y, data_new_x = load_data()

    # 2. Checking the data
    # check_data(data_x, data_y, data_new_x)

    # We can see that the DataX file has 1494 entries and 35 variables, 218 of those rows have missing values.
    # DataNewX file has 654 entries and 35 variables, 87 of those rows have missing values.
    # DataY file has 1494 entries and 2 variables, none of those rows have missing values.

    # DataX and DataY will be the training sets, and DataNewX will be the test set.

    # The ratio of rows missing values is:
    # DataX: 218/1494 = 0.146
    # DataNewX: 87/654 = 0.133

    # Since those values are low we can remove the rows with missing values.

    data_x, data_y, data_new_x = remove_missing_values(data_x, data_y, data_new_x)
    # check_data(data_x, data_y, data_new_x)

    # In our case, the target variable to be predicted in this project is the daily variation of future prices.
    # Let's apply an Explorative Data Analysis (EDA) to the data.

    # EDA
    dataframe = data_x.drop('ID', axis=1)
    dataframe = data_x.drop('DAY_ID', axis=1)
    dataframe = data_x.drop('COUNTRY', axis=1)

    dataframe.plot(kind='box', subplots=True, layout=(10, 10), sharex=False, sharey=False)
    plt.title('Box plots of the dataframe')
    plt.gcf().set_size_inches(20, 20)
    plt.subplots_adjust(wspace=0.5, hspace=0.7)
    plt.show()
    #overview_data(data_x)
    #overview_data(data_new_x)
    #overview_data(data_y)


if __name__ == '__main__':
    main()