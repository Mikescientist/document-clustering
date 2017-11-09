# Document Clustering

Here we will use data collected from Companies House from businesses in an attempt to use some unsupervised learning to uncover common topics and themes. We will encode the documents into a vector space using doc2vec - an extension of the well known word2vec vectorisation.

Once we have encoded the data, we will use some dimensionalty reduction techniques such as multidimensional scaling and t-distributed stochastic neighbour embedding to visualise the data in 2D or 3D in the hope we see some clusters starting to emerge.

Finally we will use some clustering algorithms to attempt to group common topics together. We will use a crude k-means approach as well as the more sophisticated density-based clustering of HDBScan. 
