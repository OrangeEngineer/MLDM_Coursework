import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def ScatterPlot(data1,features_mean,df):
    # Color Labels - 0 is benign and 1 is malignant
    color_dic = {0: 'blue', 1: 'green', 2: 'red'}
    target_list = list(df['GameTier'])
    colors = list(map(lambda x: color_dic.get(x), target_list))
    # Plotting the scatter matrix
    sm = pd.plotting.scatter_matrix(data1[features_mean], c=colors, alpha=0.4, figsize=((10, 10)))

    for ax in sm.flatten():
        ax.set_xlabel(ax.get_xlabel(), fontsize=8, rotation=90)
        ax.set_ylabel(ax.get_ylabel(), fontsize=8, rotation=0, )

    [s.get_yaxis().set_label_coords(-1, 0.5) for s in sm.reshape(-1)]
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.suptitle("Scatter matrix")
    plt.show()

def HeatMap(data,data_feature_names,df):
    # Arrange the data as a dataframe
    data1 = data
    data1.columns = data_feature_names
    # Plotting only 7 features out of 30
    NUM_POINTS = 10
    features_mean = list(data1.columns[0:NUM_POINTS + 3])
    print(features_mean)
    feature_names = data_feature_names[0:NUM_POINTS + 3]
    print(feature_names)
    f, ax = plt.subplots(1, 1)  # plt.figure(figsize=(10,10))
    sns.heatmap(data1[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
    # Set number of ticks for x-axis
    ax.set_xticks([float(n) + 0.5 for n in range(NUM_POINTS)])
    # Set ticks labels for x-axis
    ax.set_xticklabels(feature_names, rotation=25, rotation_mode="anchor", fontsize=10)
    # Set number of ticks for y-axis
    ax.set_yticks([float(n) + 0.5 for n in range(NUM_POINTS)])
    # Set ticks labels for y-axis
    ax.set_yticklabels(feature_names, rotation='horizontal', fontsize=10)
    plt.title("Correlation between various features")
    plt.show()
    plt.close()

    ScatterPlot(data1, features_mean,df)

def main():
    df = pd.read_csv(r'/VideoGames/VideGames.csv')
    bin_labels = [0,1,2]
    df['GameTier'] = pd.qcut(df['Score'], q=3, labels = bin_labels)
    data =  df.iloc[:,2:df.columns.get_loc('User_normalised_by_year')]
    print(df)

    df.to_csv('Preprocessed_VideoGames.csv')

    data_feature_names = [data.columns[i] for i in range(0,data.shape[1])]

    HeatMap(data,data_feature_names,df)

if __name__ == "__main__":
    main()

