# A multivariate gaussian neural network
This repo implements a [multivariate gaussian](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) 
neural network and works by seeing how close in terms of the L2 norm the samples are to a query vector. 
To generate the distributions the average vector is first calculated, and then the covariance matrix is produced. 
A standard deviation matrix is produced by taking the sqaure root of the covariance matrix. The average 
vector and standard deviation matrix form the model.
## Implementation of the iris data set
The iris data set has 4 feature per flower with three types of flowers. The data set is divided up in terms of 
flower type creating 3 sets with 50 samples each. The 3 sets of samples are used to create 3 multivariate 
gaussians. To classify a query vector of 4 measurements each multivariate gaussian is sampled from. The L2 norm is 
used to calculate which multivariate guassian produces samples closest to the query vector.
