import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def ScatterPlot(data1,features_mean,df):
 #Color Labels - 0 is benign and 1 is malignant
 color_dic = {0:'green', 1:'yellow',2:'red'}
 target_list = list(df['CrimeTier'])
 colors = list(map(lambda x: color_dic.get(x), target_list))
 #Plotting the scatter matrix
 sm = pd.plotting.scatter_matrix(data1[features_mean], c= colors, alpha=0.4, figsize=((10,10)))
 plt.suptitle("Scatter matrix")
 plt.show()

def HeatMap(data,data_feature_names,df):
 # Arrange the data as a dataframe
 data1 = data
 data1.columns = data_feature_names
 # Plotting only 7 features out of 30
 NUM_POINTS = 11
 features_mean = list(data1.columns[1:NUM_POINTS])
 feature_names = data_feature_names[1:NUM_POINTS]
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

 ScatterPlot(data1,features_mean,df)

def main():

 df = pd.read_csv(r'~/Workspace/ML_CW/LondonCrimes/MPS_Borough_Level_Crime_Historic.csv')

 MajCat, unique = pd.factorize(df['Major Category'],sort = False)
 df.insert(3, "MajorCategory",MajCat, True)

 #for category type to be a factor
 MinCat, unique = pd.factorize(df['Minor Category'],sort=False)
 df.insert(3, "MinorCategory",MinCat, True)

 year_sum = df.iloc[:,df.columns.get_loc('2008'):df.columns.get_loc('2017')].sum(axis = 1)
 normalized_year_sum=(year_sum-year_sum.mean())/year_sum.std()
 df.insert(df.shape[1], "NormalizedSumYears",normalized_year_sum, True)

 bin_labels = [0,1,2]
 df['CrimeTier'] = pd.qcut(normalized_year_sum, q=3, labels = bin_labels)
 data =  df.iloc[:,df.columns.get_loc('2008'):df.shape[1]-1]
 print(df.keys())

 data_feature_names = [data.columns[i] for i in range(0,data.shape[1])]

 df.to_csv('Preprocessed_LodonCrime.csv')

 HeatMap(data,data_feature_names,df)


if __name__ == "__main__":
    main()