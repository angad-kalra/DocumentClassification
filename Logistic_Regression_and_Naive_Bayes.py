from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
with open("farm-ads.txt","r") as myfile:
	data = myfile.readlines()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)	#.todense())
X = X.toarray()						#necessary as the COO (co-ordinate) type matrix causes issues in dot product calculation
print ("X shape: ",X.shape)

#defining label vector
y_data = np.loadtxt("farm-ads-label.txt",skiprows=0, unpack=True)
y = np.asarray(y_data,int)
Y = y[1]
print("Y shape:\t",Y.shape)
#print (Y)
#print (X.shape)							#matrix of shape 4143x43602



#Logistic Regression code
#Red color for LR
#finding log likelihood
def log_likelihood(X_feat, Y_label, weights):
	w_x_dot_prod = np.dot(X_feat,weights)
	log_like = np.sum( (Y_label - 1)*w_x_dot_prod - np.log( 1 + np.exp(w_x_dot_prod) ) )
	return log_like


def sigmoid(s):
	#print ("dot prod: ",s.shape)
	return ( 1 / (1 + np.exp(-s)) )


def gradient_log_likelihood(X_feat, Y_label, no_of_steps, learning_rate, add_intercept=False):
	if(add_intercept==True):
		#code for adding intercept 						
		inter = np.ones((X_feat.shape[0],1))
		#print(inter.shape)
		X_feat = np.hstack((inter, X_feat))
	
	#print("X-feat shape:\t",X_feat[1].shape)
	
	weights = np.zeros(X_feat.shape[1])
	
	#print ("weights shape:\t",weights.shape)
	
	for step in range(no_of_steps):
		w_x_dot_prod = np.dot(X_feat,weights)
		prediction = sigmoid(w_x_dot_prod)
		#print ("here\t", prediction.shape)
		
		#Updating weight vector with gradient
		output_error = Y_label - prediction
		#print ("oe shape:\t",output_error.shape)

		gradient = np.dot(X_feat.T, output_error)
		#print("gradient shape:\t",gradient.shape)
		
		weights += learning_rate*gradient
		
		#print("hello ")
		
		#Print log likelihood at some intervals
		if (step % 10000 == 0):
			print (log_likelihood(X_feat,Y_label,weights))
	
	return weights





#Accuracy logic
#data_with_intercept = np.hstack((np.ones((X.shape[0], 1)),X))
size = [0.9,0.7,0.5,0.3,0.2,0.1]
accuracy = [0,0,0,0,0,0]

for i in range(len(size)):
        temp = 0
        for g in range(5):
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size[i]) # random_state = 0)
                weights = gradient_log_likelihood(X_train, Y_train, 400, 5e-5, True)
                X_test_w_intercept = np.hstack((np.ones((X_test.shape[0], 1)),X_test))
                final_scores = np.dot(X_test_w_intercept, weights)
                preds = np.round(sigmoid(final_scores))
                temp = temp + (((preds == Y_test).sum().astype(float) / len(preds))*100)
        accuracy[i] = temp/5    #((preds == Y_test).sum().astype(float) / len(preds))*100
        print ("Accuracy from scratch: {0}".format(accuracy[i]))

plt.xlabel('Train size')
plt.ylabel('Accuracy')
for i in range(len(size)):
        size[i] = 1- size[i]
plt.plot(size,accuracy,'r')
#plt.show()




#Naive bayes below
#Blue color for NB
class Multinomial_Naibe_Bayes(object):

	def __init__(self,alpha = 1.0):
		self.alpha = alpha

	def calc_probabilities(self, X, Y):
		separated = [[attr_set for attr_set,t in zip(X,Y) if t==c] for c in np.unique(Y)]
		count_sample = X.shape[0]
		self.log_class_prior = [np.log(len(i)/count_sample) for i in separated]				#prior log probability for each class
		words_count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha	#count of each word for each class
		#print("this ",words_count.shape)
		#print("this too ",words_count.sum(axis=1))
		self.log_likelihood = np.log((words_count)/(words_count.sum(axis=1)+43602)[np.newaxis].T)	#add laplace smoothing
		return self

	def predict_log_prob(self,X):
		temp = ([(self.log_likelihood * x).sum(axis=1) + self.log_class_prior for x in X])
		return temp

	def predict_max(self, X):
		tmp2 = np.argmax(self.predict_log_prob(X), axis=1)				#calls predict_log probabilities and picks maximum value
		print("he\n",tmp2.shape)
		return tmp2

	def score(self, X, Y):
		return (sum(self.predict_max(X) == Y)/len(Y))




size_nb = [0.9,0.7,0.5,0.3,0.2,0.1]
accuracy_nb = [0,0,0,0,0,0]
for i in range(len(size)):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size_nb[i], random_state = 0)
        nb = Multinomial_Naibe_Bayes().calc_probabilities(X_train, Y_train)
        accuracy_nb[i] = nb.score(X_test, Y_test)*100
        print(accuracy_nb[i])

#plt.xlabel('Train size')
#plt.ylabel('Accuracy')
for i in range(len(size_nb)):
        size_nb[i] = 1- size_nb[i]
plt.plot(size_nb,accuracy_nb,'b')

plt.show()
