import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Function to load data
@st.cache(allow_output_mutation=True)
def load_data():
    # Adjust the path to where you will upload your dataset in the deployment environment
    data = pd.read_json('dataset.json', lines=True)
    return data

# Function to perform TF-IDF transformation
@st.cache(allow_output_mutation=True)
def tfidf_transform(data):
    tfidf = TfidfVectorizer(
        min_df=5,
        max_df=0.95,
        max_features=8000,
        stop_words='english'
    )
    text = tfidf.fit_transform(data['contents'])
    return text, tfidf

# Function to find optimal clusters
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1, 2)
    sse = []
    silhouette_scores = []
    
    for k in iters:
        model = MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20)
        labels = model.fit_predict(data)
        sse.append(model.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))
        st.write(f'For k={k}, Silhouette Score:{silhouette_score(data, labels)}')  # Display silhouette score in the app
        
    return iters, sse, silhouette_scores

# Function to plot TSNE and PCA
def plot_tsne_pca(data, labels):
    # Your existing code for plotting
    max_label = max(labels)
    max_items_size = min(300, data.shape[0])
    max_items = np.random.choice(range(data.shape[0]), size=min(300, data.shape[0]), replace=False)

    data_array = data[max_items, :].toarray()
    pca = PCA(n_components=2).fit_transform(data_array)
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data_array))

    idx = np.random.choice(range(pca.shape[0]), size=min(300, pca.shape[0]), replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')

# Function to get top keywords for each cluster
def get_top_keywords(data, clusters, vectorizer, n_terms):
    # Your existing code for getting top keywords
    feature_names = np.array(vectorizer.get_feature_names_out())
    df = pd.DataFrame(data.todense(), columns=feature_names)

    for i in range(max(clusters) + 1):
        cluster_data = df[clusters == i]
        mean_values = cluster_data.mean(axis=0)
        top_terms = mean_values.argsort()[-n_terms:][::-1]
        keywords = ', '.join([feature_names[t] for t in top_terms])

        st.write('\nCluster {}: {}'.format(i, keywords))

# Main function to orchestrate the Streamlit app
def main():
    st.title('Text Clustering App')

    # Load data and perform TF-IDF transformation
    data = load_data()
    text, tfidf = tfidf_transform(data)

    # User inputs
    st.sidebar.title("Configuration")
    num_clusters = st.sidebar.slider('Number of clusters', 2, 20, 5)
    n_terms = st.sidebar.slider('Number of top keywords', 5, 20, 10)

    # Button to calculate optimal clusters
    if st.sidebar.button('Find Optimal Clusters'):
        st.title('Silhouette Score')
        iters, sse, silhouette_scores = find_optimal_clusters(text, 20)
        fig, ax1 = plt.subplots()  # Create a figure and an Axes object
        ax1.plot(iters, sse, marker='o')  # Use ax1 here to plot
        ax1.set_xlabel('Cluster Centers')
        ax1.set_ylabel('SSE')
        ax1.set_title('SSE by Cluster Center Plot')
        st.pyplot(fig)  # Display the figure

    # Button to perform clustering and show results
    if st.sidebar.button('Cluster Data'):
        clusters = MiniBatchKMeans(n_clusters=num_clusters, init_size=1024, batch_size=2048, random_state=20).fit_predict(text)
        plot_tsne_pca(text, clusters)  # This function creates its own figure and doesn't return it
        st.pyplot(plt.gcf())  # Display the current figure

        get_top_keywords(text, clusters, tfidf, n_terms)

if __name__ == '__main__':
    main()