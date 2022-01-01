import pandas as pd
import numpy as np
import scipy
from plotly.subplots import make_subplots
import plotly.express as px


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import classification_report

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(filename):
    """Load data from file.

    Parameters
    ----------
    filename : path and name of data table file in excel format.

    Returns
    -------
    df : dataframe
    """
    try:
        df = pd.read_excel(filename)
        return df
    except:
        print('Check path and only read excel file.\n')
    
def inspect_data(df):
    """Data inspection: 
        Show data's first 5 rows, data's info (null-values, dimensions, types...)

    Parameters
    ----------
    df : dataframe of which labels are in the last column.     
    """
    print('\n*******\nData Inspection\n')
    print(df.head())
    print(df.info())

def boxplot_features(df):
    """Plot statistical distribution of each data feature in the dataframe.
        By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    df : dataframe
    """
    df1 = pd.DataFrame(df.iloc[:,:-1])
    fig = px.box(df1.melt(), y="value", facet_col="variable", boxmode="overlay", color="variable")
    fig.update_yaxes(showticklabels=True, matches=None)
    fig.show()

def countplot_target(df):
    """Plot histogram of each data feature in the dataframe.
        By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    df : dataframe
    """
    fig = px.histogram(df, x = 'Class', color='Class', title='Numbers of each labels')
    fig.show()

def encode_label(df):
    """Encode labels of the features in the dataframe.

    Parameters
    ----------
    df : dataframe
    """
    df.iloc[:,-1] = LabelEncoder().fit_transform(df.iloc[:,-1])
    print('\n*******\nData has been encoded.\n')
    return df

def scale_data(df):
    """Encode labels and standardize data's features.
        Return transformed arrays of features, of labels of the data.

    Parameters
    ----------
    df : dataframe
        labels/targets are in the last column.

    Returns
    -------
    X : transformed array-like of shape (n_samples, n_features)
    y : transformed array-like of shape (n_samples,)
    """
    y = df.iloc[:,-1]
    X = StandardScaler().fit_transform( df.iloc[:,:-1])
    print('\n*******\nData has been scaled.\n')
    return X, y

def run_PCA(X, y, n_components = 3):
    """Implement principle component analysis (PCA) on data.
        Plot the PCA-analysed results. By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        `n_samples` is the number of samples
        `n_features` is the number of features.
    
    y : array-like of shape (n_samples,)   

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
    plot_data(X_embedded, y, title = f'Data before clustering: PCA {n_components}  components, show max 3 components.\n')
    return X_embedded
    
def plot_cumulative_explained_variance(X):
    """Plot PCA cumulative explained variance.
    By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        `n_samples` is the number of samples
        `n_features` is the number of features.
    """
    print('\n*******\nPlot cumulative explained variance (PCA)\n')
    pca = PCA()
    pca.fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    df_variance = pd.DataFrame({'Number of components': range(1,len(cum_var)+1), 'Cumulative explained variance': cum_var})
    fig = px.line(df_variance, x ='Number of components', y ='Cumulative explained variance', title='Cumulative explained variance (PCA)')
    fig.show()    

def run_2D_tSNE(X, y, perplexity = 25):
    """Implement T-distributed Stochastic Neighbor Embedding (n_components = 2) with 
        perplexity between 0 and 50 on data.
        Plot the result. By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        `n_samples` is the number of samples
        `n_features` is the number of features.

    y : array-like of shape (n_samples,)

    perplexity : int between 0 and 50, default = 25

    Returns
    -------
    X_embedded : transformed array-like of shape (n_samples, n_features)
    """
    try:
        range(1,52)[perplexity]
        tsne = TSNE(n_components=2, perplexity= perplexity)
        X_embedded = tsne.fit_transform(X)
        print(f'\nRun tSNE - n_components = 2 - perplexity = {perplexity}')
        plot_data(X_embedded, y, title = f'tSNE 2 components - perplexity {perplexity}')
        return X_embedded
    except:
        print('Perplexity should be an integer between 0 to 50.\n')      

def plot_data(X, y, title, cluster_centers=None, n_clusters=None):
    """Display data on the plot, show maximum 3 dimensions. By default figure is opened in a tab of the default web browser.
        
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        `n_samples` is the number of samples
        `n_features` is the number of features.

    y : array-like of shape (n_samples,)

    title : title string

    cluster_centers : coordinates of cluster centers, ndarray of shape (n_clusters, n_features), default = None

    n_clusters : target (positive integer) number of clusters, default = None
    """
    print(f'\n*******\nPlot data: {title}')   
    if X.shape[1] == 1:
        d3 = 0
        d2 = 0
        d1 = X[:, 0]
        if cluster_centers is not None:
            c1 = cluster_centers[:,0]
            c2 = np.zeros(n_clusters)
            c3 = np.zeros(n_clusters)
    elif X.shape[1] == 2: 
        d3 = 0
        d2 = X[:, 1]
        d1 = X[:, 0]
        if cluster_centers is not None:
            c1 = cluster_centers[:,0]
            c2 = cluster_centers[:,1]
            c3 = np.zeros(n_clusters)
    else:
        d3 = X[:, 2]
        d2 = X[:, 1]
        d1 = X[:, 0]
        if cluster_centers is not None:
            c1 = cluster_centers[:,0]
            c2 = cluster_centers[:,1]
            c3 = cluster_centers[:,2]
    df_plot = pd.DataFrame({'Dimension one': d1, 'Dimension two': d2, 'Dimension three': d3, 'color': y})
    fig = px.scatter_3d(df_plot, x='Dimension one', y='Dimension two', z='Dimension three', color='color', title = title)
    if cluster_centers is not None:
        fig.add_scatter3d(x=c1, y=c2, z=c3, mode='markers', marker=dict(color='lightgreen',size=10))
    fig.show()

def run_Kmeans(X, y, n_clusters, title):
    """Run K-Means clustering algorithm on data.
        Display the average Silhoutte score. Display classification report if n_clusters is 7.
        Plot the clustering results. By default figure is opened in a tab of the default web browser.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        `n_samples` is the number of samples
        `n_features` is the number of features.

    y : array-like of shape (n_samples,)

    n_clusters : target (positive integer) number of clusters

    on_data : string variable to indicate whether the input data is one of those in the set 
        {'scaled data', 'the selected n_components PCA data','the 2 components tSNE data'}. 
    """
    print(f'\n*******\n{title}\n')
    model = KMeans(n_clusters = n_clusters).fit(X)
    y_pred = model.predict(X)
    silhouette_avg = silhouette_score(X,model.labels_)
    davies_bouldin = davies_bouldin_score(X,model.labels_)
    print(f'\nThe average silhoutte score is: {silhouette_avg:.3f}\n')
    print(f'The davies bouldin score is: {davies_bouldin:.3f}\n')
    plot_data(X, y_pred, title, model.cluster_centers_, n_clusters)
  
    if n_clusters == len(set(y)):
        try:
            permutation = []
            for i in range(n_clusters):
                new_label = scipy.stats.mode(y_pred[y==i])[0][0] 
                permutation.append(new_label)
            new_y = [permutation[i] for i in y]
            print(f'{classification_report(new_y, y_pred, zero_division=1)}\n')
        except:
            pass

def test_KMeans_2_to_9_clusters(X, title):
    """Run K-Means clustering algorithm on data.
        Display average Silhoutte scores, davies_bouldin scores of models for each number of cluster (2 to 9 clusters) uses.
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
    for i in range(2,10):
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
    df_elbow = pd.DataFrame({'Number of clusters': range(2,10), 'Sum of inertia': sse})
    fig = px.line(df_elbow, x='Number of clusters', y = 'Sum of inertia', title='Elbow Method')
    fig.show()
   

def main():
    """ Main block of code."""
    print('\n**************\nSTART THE PROGRAM ...')
    filename = 'C:/Users/lynguyen/Documents/Study/EC/Visualization/Inlämningsuppgift_LyNguyen/DryBeanDataset/Dry_Bean_Dataset.xlsx'
    
    #load data
    df = load_data(filename)
    
    #inspect data
    inspect_data(df)

    #boxplot all features
    boxplot_features(df)

    #encode data labels for visualization
    df = encode_label(df)

    #count plot targets
    countplot_target(df)
    
    #choose n_clusters
    n_clusters= 7

    #run Kmeans with selected clusters on scaled data
    X_scaled, y = scale_data(df)
    run_Kmeans(X_scaled, y, n_clusters, title=f'Clustering with K-Means {n_clusters} clusters on scaled data')
   
    #plot PCA cumulative explained variance on scaled data
    plot_cumulative_explained_variance(X_scaled)

    #choose PCA n_components
    PCA_components=5

    #run PCA selected n_components
    X_embedded = run_PCA(X_scaled, y, n_components = PCA_components)  

    #run Kmeans on PCA data: test from 2 to 9 clusters 
    test_KMeans_2_to_9_clusters(X_embedded, title='Clustering with 2 to 9 clusters K-Means on {PCA_components} components PCA data')
    
    #choose perplexity
    perplexity = 50

    #run tSNE 2 components, selected perplexity on scaled data
    X_embedded = run_2D_tSNE(X_scaled, y, perplexity)

    #run Kmeans selected n_clusters on tSNE data
    run_Kmeans(X_embedded, y, n_clusters, title=f'Clustering with K-Means {n_clusters} clusters onthe 2 components tSNE data')
    print('\n**************\n... END.')
   
    
if __name__=="__main__":
    main()