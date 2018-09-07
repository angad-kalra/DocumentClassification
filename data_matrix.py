from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.io
with open("farm-ads.txt","r") as myfile:
	data = myfile.readlines()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)	#.todense())
#X = x.toarray()						#necessary as the COO (co-ordinate) type matrix causes issues in dot product calculation
X = X.toarray()
print ("X shape: ",X.shape)				#indexed as X[<row>,<column>]

#print (X.shape)							#matrix of shape 4143x43602
x_t= X.transpose()
scipy.io.savemat('q6a.mat', mdict={'x':x_t})