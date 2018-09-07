# DocumentClassification
Each line in data file represents a text used for farm advertisement. All the stop words have been removed from the texts.  In the label file, the label +1 means the corresponding ad is accepted, while the label 0 means the ad is rejected.</br>
For this I have implemented both Logistic Regression and Naive Bayes from scratch, without using any existing packages and functions to predict whether an ad can be accepted or not.</br></br>
NOTE: I used the bag-of-words model and the number of occurrence of words in each ad as its feature. I assume that the positions of words do not matter and each attribute value is independently generated.
