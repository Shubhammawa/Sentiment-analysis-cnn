# Sentiment-analysis-cnn
Text classification using Convolutional Neural Networks

# Dataset Description
The dataset contains reviews from the following 3 websites,
Amazon, Imdb, Yelp.
There are 1000 reviews for each website, 500 of which are positive.
In total we have 1500 positive reviews and 1500 negative reviews.

# Model Description
The model takes inspiration from the paper, "Sentence Classification using Convolutional Neural Networks" by Yoon Kim.
[Paper](https://arxiv.org/abs/1408.5882)

Kim CNN:
![Kim CNN](https://github.com/Shubhammawa/Netapp-Data-Challenge-Kshitij/blob/master/KimCNN.png)
# Hyperparameter Tuning
Tunable hyperparameters:
1. Word vector size (embedding size)
2. Sequence length  (after padding or truncation)
3. Filter sizes
4. Number of filters of each type
5. Learning rate
6. Num_epochs
7. minibatch_size 
