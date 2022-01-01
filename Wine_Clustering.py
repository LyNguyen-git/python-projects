import pandas as pd
import numpy as np
import scipy
from plotly.subplots import make_subplots
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import classification_report


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data():
    """Load data from urls.

    Returns
    -------
    df : dataframe
    """
    try:
        df_red = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
        df_white = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
        df_red['wine type'] = 'red'
        df_white['wine type'] = 'white'
        df = pd.concat([df_red, df_white], axis = 0)
        return df
    except:
        print('Broken links, please read directly from folder.\n')
    
def inspect_data(df):
    """Data inspection: 
        show data's first 5 rows, data's info (null-values, dimensions, types...)
        Count plot targets. By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    df : dataframe    
    """
    print('\n*******\nData Inspection\n')
    print(df.head())
    print(df.info())
    countplot_target(df,title='Numbers of quality at beginning')
   
def get_data(df, wine_color):
    """Get data of red or white wine.

     Parameters
    ----------
    df : dataframe  

    wine_color : string, "red" or "white"

    Returns
    -------
    df : dataframe
    """
    print(f'\n*******\nData {wine_color} wine has been chosen.\n')
    return df[df['wine type']== wine_color]

def countplot_target(df, title):
    """Plot histogram of each target label in the dataframe.
        By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    df : dataframe

    title : title string 
    """
    fig = px.histogram(df, x ='quality', color='wine type', barmode='group', opacity=0.8,
                        color_discrete_map={'red': 'red', 'white': 'lightyellow'}, title=title)
    fig.show()
    
def rearrange_target(df):
    """Re-arrange target labels in the dataframe.
        Count plot new targets. By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    df : dataframe

    Returns
    -------
    df : dataframe with new target labels
    """
    df['quality'] = df['quality'].astype(int).apply(lambda i: 3 if i <=5 else 2 if i <7 else 1)
    print('\n*******\nQuality has been re-arranged from level 3 to 1.\n')
    countplot_target(df,title='Numbers of quality after re-arrangement')
    return df

def boxplot_features(df, title):
    """Plot statistical distribution of each data feature in the dataframe.
        By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    df : dataframe

    title : title string 
    """
    print('\n*******\nPlot statistical distribution of each data feature in the dataframe.\n')
    features = [f for f in df.columns if f not in ['quality','wine type']]
    df_plot = pd.melt(df, id_vars=['quality', 'wine type'], value_vars=features)
    
    # plot the full (non-cluster) data
    fig = px.box(df_plot, title = title, y='value', facet_col='variable', boxmode='overlay', color='wine type',
                color_discrete_map={'red': 'red', 'white': 'lightgoldenrodyellow'})
    fig.update_yaxes(showticklabels=True, matches=None)
    fig.show()

    # plot per-class data
    fig = px.box(df_plot, title = title, y='value', x= 'quality', facet_col='variable', boxmode='overlay', color='wine type',
                 color_discrete_map={'red': 'red', 'white': 'lightgoldenrodyellow'})
    fig.update_yaxes(showticklabels=True, matches=None)
    fig.show()

def plot_feature_distribution_after_clustering(df, y_pred, title):
    """Plot statistical distribution of each data feature in the selected wine color data after clustering.
        By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    df : dataframe

    y_pred : the result of each sample's clustering assignment

    title : title string 

    """
    print(f'\n*******\n{title}.\n')
    df['quality'] = y_pred
    boxplot_features(df, title)

def scale_data(df):
    """Standardize data's features.
        Return transformed arrays of features, of labels of the data.

    Parameters
    ----------
    df : dataframe

    Returns
    -------
    X : transformed array-like of shape (n_samples, n_features)
    y : transformed array-like of shape (n_samples,)
    """
    y = df['quality']
    X= StandardScaler().fit_transform(df.drop('wine type', axis=1))
    print(f'\n*******\nData has been scaled.\n')
    return X, y

def run_PCA(X, y, title, n_components=3):
    """Implement principle component analysis (PCA) on data.
        Plot the PCA-analysed results.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        `n_samples` is the number of samples
        `n_features` is the number of features.
    
    y : array-like of shape (n_samples,)   

    title : title string 

    n_components : integer (PCA component), default= 3

    Returns
    -------
    X_embedded : transformed array-like of shape (n_samples, n_features)
    """
    pca = PCA(n_components = n_components)
    pca.fit(X)
    X_embedded = pca.transform(X)
    if n_components == None:
        print(f'\n*******\nRun PCA - n_components = Full\n')
    else: 
        print(f'\n*******\nRun PCA - n_components = {n_components}\n')
    plot_data(X_embedded, y, title)
    return X_embedded
    
def plot_cumulative_explained_variance(X, title):
    """Plot PCA cumulative explained variance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        `n_samples` is the number of samples
        `n_features` is the number of features.
    
    title : title string 
    """
    print(f'\n*******\n{title}\n')
    pca = PCA()
    pca.fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    df_variance = pd.DataFrame({'Number of components': range(1,len(cum_var)+1), 'Cumulative explained variance': cum_var})
    fig = px.line(df_variance, x ='Number of components', y ='Cumulative explained variance', title=title)
    fig.show()    


def plot_data(X, y, title):
    """Display data on the plot, show maximum 3 dimensions.
        
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        `n_samples` is the number of samples
        `n_features` is the number of features.

    y : array-like of shape (n_samples,)

    title : title string
    """
    print(f'\n*******\nPlot data: {title}')   
    if X.shape[1] == 1:
        d3 = 0
        d2 = 0
        d1 = X[:, 0]
    elif X.shape[1] == 2: 
        d3 = 0
        d2 = X[:, 1]
        d1 = X[:, 0]   
    else:
        d3 = X[:, 2]
        d2 = X[:, 1]
        d1 = X[:, 0]       
    df_plot = pd.DataFrame({'Dimension one': d1, 'Dimension two': d2, 'Dimension three': d3, 'color': y})
    fig = px.scatter_3d(df_plot, x='Dimension one', y='Dimension two', z='Dimension three', color='color', title = title)
    fig.show()

def run_Agglomerative(X, n_clusters, title):
    """Run Agglomerative clustering algorithm on data.
        Display average Silhoutte scores, davies_bouldin scores of the model.
        Plot the clustering results.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        `n_samples` is the number of samples
        `n_features` is the number of features.

    n_clusters : target (positive integer) number of clusters

    title : title string 
    """
    print(f'\n*******\n{title}\n')
    model = AgglomerativeClustering(n_clusters, linkage='ward', compute_full_tree=True).fit(X)
    y_pred = model.fit_predict(X)
    silhouette_avg = silhouette_score(X,model.labels_)
    davies_bouldin = davies_bouldin_score(X,model.labels_)
    print(f'\nThe average silhoutte score is: {silhouette_avg:.3f}')
    print(f'The davies bouldin score is: {davies_bouldin:.3f}\n')
    plot_data(X, y_pred, title)
    return y_pred

def test_KMeans_2_to_7_clusters(X, title):
    """Run K-Means clustering algorithm on data.
        Display average Silhoutte scores, davies_bouldin scores of models for each number of cluster (2 to 7 clusters).
        Display the number of cluster which gives best average Silhoutte score, davies_bouldin score.
        Plot the elbow method. By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        `n_samples` is the number of samples
        `n_features` is the number of features.

    title : string
    """
    print(f'\n*******\n{title}\n')
    sse=[]
    best_silhouette =-1
    best_settings_silhouette = None
    silhouette=[]
    daviesbouldin={}
    for i in range(2,8):
        model = KMeans(n_clusters = i).fit(X)
        sse.append(model.inertia_)
        silhouette_avg = silhouette_score(X,model.labels_)
        davies_bouldin = davies_bouldin_score(X,model.labels_)
        silhouette.append(silhouette_avg)
        daviesbouldin[i] = davies_bouldin
        
        print(f'For n_clusters = {i} - The average silhoutte_score: {silhouette_avg:.3f} - The davies_bouldin score: {davies_bouldin:.3f} ')  
        if silhouette_avg > best_silhouette:
            best_settings_silhouette = {"Clusters": i, "Score": silhouette_avg}
            best_silhouette = silhouette_avg
    print(f'\nBest number of cluster, the average silhoutte score: {list(best_settings_silhouette.values())[0]}, {list(best_settings_silhouette.values())[1]:.3f}\n')
    print(f'Best number of cluster, the davies bouldin score: {min(daviesbouldin,key=daviesbouldin.get)}, {min(daviesbouldin.values()):.3f}')

    print('\nPlot Elbow Method')
    df_elbow = pd.DataFrame({'Number of clusters': range(2,8), 'Sum of inertia': sse})
    fig = px.line(df_elbow, x='Number of clusters', y = 'Sum of inertia', title=f'Elbow Method - {title}')
    fig.show()
   
    
def test_Agglomerative_2_to_7_clusters(X, title):
    """Run Agglomerative clustering algorithm on data.
        Display average Silhoutte scores, davies_bouldin scores of models for each number of cluster (2 to 7 clusters).
        Display the number of cluster which gives best average Silhoutte score, davies_bouldin score.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        `n_samples` is the number of samples
        `n_features` is the number of features.

    title : string
    """
    print(f'\n*******\n{title}\n')
    best_silhouette =-1
    best_settings_silhouette = None
    silhouette=[]
    daviesbouldin={}
    for i in range(2,8):
        model = AgglomerativeClustering(n_clusters=i, compute_full_tree=True).fit(X)
        silhouette_avg = silhouette_score(X,model.labels_)
        davies_bouldin = davies_bouldin_score(X,model.labels_)
        silhouette.append(silhouette_avg)
        daviesbouldin[i] = davies_bouldin
        
        print(f'For n_clusters = {i} - The average silhoutte_score: {silhouette_avg:.3f} - The davies_bouldin score: {davies_bouldin:.3f} ')  
        if silhouette_avg > best_silhouette:
            best_settings_silhouette = {"Clusters": i, "Score": silhouette_avg}
            best_silhouette = silhouette_avg
    print(f'\nBest number of cluster, the average silhoutte score: {list(best_settings_silhouette.values())[0]}, {list(best_settings_silhouette.values())[1]:.3f}\n')
    print(f'Best number of cluster, the davies bouldin score: {min(daviesbouldin,key=daviesbouldin.get)}, {min(daviesbouldin.values()):.3f}')


def main():
    """ Main block of code."""
    print('\n**************\nSTART THE PROGRAM ...')
    
    #load data
    data = load_data()

    #inspec data: first 5 rows, data info (null-value, types, shape, ...)
    inspect_data(data)

    #re-arrange targets from 7 qualities to 3 qualities for visualization
    df_rearranged = rearrange_target(data)
    boxplot_features(df_rearranged, title='Plot statistical distribution of each feature in the data')

    #select wine and get data for clustering
    wine_color = 'red'
    df = get_data(df_rearranged, wine_color)

    #scale selected data
    X_scaled, y = scale_data(df)  

    #plot PCA cumulative expained variance
    plot_cumulative_explained_variance(X_scaled, title=f'Cumulative explained variance (PCA)- {wine_color} wine data')

    #choose n_clusters
    n_clusters= 3

    #run Agglomerative with selected n_clusters on scaled data
    y_pred = run_Agglomerative(X_scaled, n_clusters, title=f'Clustering with Agglomerative {n_clusters} clusters on scaled - {wine_color} wine data')

    #plot feature distributions after clustering
    if n_clusters == 3: 
        plot_feature_distribution_after_clustering(df, y_pred, title=f'Feature distributions after clustering - {wine_color} wine data')
    
    #choose PCA n_components
    PCA_components=8
    
    #run PCA on scaled data with selected components
    X_embedded = run_PCA(X_scaled, y, title=f'Data {wine_color} wine - {PCA_components} PCA component, show max 3 PC', n_components=PCA_components)

    #run Agglomerative with selected n_clusters on PCA data
    run_Agglomerative(X_embedded, n_clusters, title=f'Clustering with Agglomerative {n_clusters} clusters on PCA {PCA_components} components - {wine_color} wine data')

    #run Agglomerative on PCA embedded data: test from 2 to 7 clusters
    test_Agglomerative_2_to_7_clusters(X_embedded, title=f'Run Agglomerative 2 to 7 clusters on PCA {PCA_components} components - {wine_color} wine data')
   
    #run Kmeans on PCA embedded data: test from 2 to 7 clusters
    test_KMeans_2_to_7_clusters(X_embedded, title=f'Run Kmeans 2 to 7 clusters on PCA {PCA_components} components - {wine_color} wine data')
    
    print('\n**************\n...END.')
   
    
if __name__=="__main__":
    main()