# Sentiment-Analysis-CNN
Text classification using Convolutional Neural Networks

# Dataset Description
The dataset contains reviews from the following 3 websites,
Amazon, Imdb, Yelp.
There are 1000 reviews for each website, 500 of which are positive.
In total we have 1500 positive reviews and 1500 negative reviews.

Update: Added new dataset consisting of movie reviews with 5531 positive training examples and 5531 negative training examples.

# Model Description
The model takes inspiration from the paper, "Sentence Classification using Convolutional Neural Networks" by Yoon Kim.
[Paper](https://arxiv.org/abs/1408.5882)

Kim CNN:
![Kim CNN](https://github.com/Shubhammawa/Netapp-Data-Challenge-Kshitij/blob/master/KimCNN.png)

#### CONVOLUTIONAL LAYER
  * Multiple filters of varying window size convolved over each training example.
  * Each filter generates a feature map.
  * Filters with different window sizes capture context and relation between words.
#### MAX-POOLING LAYER
  * Max-pooling operation performed on each feature map to get one feature per filter.
  * The idea is to capture the most important feature necessary for classification.
  * Naturally deals with variable length sentences
#### FULLY CONNECTED LAYER
#### SOFTMAX LAYER
   * Probability distribution over labels obtained.

# Hyperparameter Tuning
Tunable hyperparameters:
1. Word vector size (embedding size)
2. Sequence length  (after padding or truncation)
3. Filter sizes
4. Number of filters of each type
5. Learning rate
6. Reguralization constant
7. Num_epochs
8. minibatch_size 

# Learning Curves
![Batch size too small](https://github.com/Shubhammawa/Sentiment-analysis-cnn/blob/master/Learning_curves/Set_7.png) ![Near optimal hyperparameters](https://github.com/Shubhammawa/Sentiment-analysis-cnn/blob/master/Learning_curves/Set_13.png)
1)Batch size too small.                    2)Near optimal hyperparameters

# Results 
![1](https://github.com/Shubhammawa/Sentiment-analysis-cnn/blob/master/Results/Results_1.png)

![2](https://github.com/Shubhammawa/Sentiment-analysis-cnn/blob/master/Results/Results_3.png)

# Instructions for use:
1. Data_prepartion.ipynb used to convert tab separated data into csv format [sentence,category].
2. Final_Code_CNN.ipynb uses csv input as mentioned above and trains the CNN model.
3. CNN_code_raw folder contains older versions of the Final_Code_CNN with various intermediate blocks to print output for better visualization, understanding and debugging. (Final_Code_CNN contains only necessary blocks to train the model.)
4. If trying to reproduce results on the same dataset, no need to run Data_preparation.ipynb, data in proper format already present in Processed_Data folder.
