from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import pickle
import timeit


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
 
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    sig_val = 1/(1 + np.exp(-z))
    return  sig_val#your code here
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    list1 = mat.keys();list1
    #print list1
    #Pick a reasonable size for validation data    
    
    #initialising all the lists to their sizes and making arrays
    train_data = np.array([])
    train_data = np.ndarray(shape=(0,784))
    train_label = np.array([])
    #train_label = np.ndarray(shape=(50000,10))
    validation_data = np.array([])
    validation_data = np.ndarray(shape=(0,784))
    validation_label = np.array([])
    test_data = np.array([])
    test_data = np.ndarray(shape=(0,784))
    test_label = np.array([])

    #initialsing all labels to 0s
    train_label = ([[0 for x in range(10)] for x in range(50000)])
    #print 'train_label shape is : ' +  str(np.shape(train_label))
    validation_label = ([[0 for x in range(10)] for x in range(10000)])
    #print 'validation_label shape is : ' +  str(np.shape(validation_label))
    test_label = ([[0 for x in range(10)] for x in range(10000)])
    #print 'test_label shape is : ' +  str(np.shape(test_label))

    train_label = np.array(train_label)
    validation_label = np.array(validation_label)
    test_label = np.array(test_label)
    
    #Your code here

    #0th axis initialisation
    traindatalist1 = np.empty([])
    traindatalist1 = mat.get('train0')            
    aperm = np.random.permutation(traindatalist1.shape[0])
    train_a1 = traindatalist1[aperm[0:5000],:]
    validation_a1 = traindatalist1[aperm[5000:],:]
    train_data = np.concatenate([train_data, train_a1], 0)
    validation_data = np.concatenate([validation_data, validation_a1], 0)
       
    train_label = np.ndarray(shape=(50000,10))   
    train_label[0:5000,0] = 1
    validation_label = np.ndarray(shape=(10000,10))   
    validation_label[0:1000,0] = 1
    test_label = np.ndarray(shape=(10000,10))   
    test_label[0:1000,0] = 1

    #initialize all test data
    for i in range(0, 10):
        test_set = 'test'+str(i)
        B = mat.get(test_set)
        test_data = np.concatenate([test_data, B], 0)
    print 'test_data size is : ' + str(test_data.shape)     
    
    #all remaining data initialisation     
    for i in range(1, 10):
        train_label[(i*5000): ((i+1)*5000),i] = 1
        validation_label[(i*1000): (i+1)*1000,i] = 1
        test_label[(i*1000): (i+1)*1000,i] = 1
    
        A = mat.get('train'+str(i))
        aperm = np.random.permutation(A.shape[0])
        A1 = A[aperm[0:5000],:]
        A2 = A[aperm[5000:],:]
        train_data = np.concatenate([train_data, A1], 0)
        validation_data = np.concatenate([validation_data, A2], 0)
    #print 'train_data 0th row', train_data[0]
    print 'train_data ', train_data.shape
        
    #feature selection
    #remove pixels whose values are same for all images across all training data
    bool_arr_train=np.all(train_data==train_data[0,:],axis=0)
    #bool_arr_validation=np.all(validation_data==validation_data[0,:],axis=0)
    #bool_arr_test=np.all(test_data==test_data[0,:],axis=0)

    #bool_arr is a boolean array
    #If an element in bool_arr is true, then the corresponding column in train_data has all its elements equal
    indices_arr_train=[]
    #indices_arr_validation=[]
    #indices_arr_test=[]
    
    for i in range(bool_arr_train.size):
        if bool_arr_train[i]==True:
            indices_arr_train.append(i)
            #indices_arr_validation.append(i)
            #indices_arr_test.append(i)

    #for i in range(bool_arr_train.size):
     #   if bool_arr_train[i]==True:
    
    #for i in range(bool_arr_train.size):
     #   if bool_arr_train[i]==True:
    
    
    #indices_arr contains the indices of the columns whose all elements are equal
    train_data=np.delete(train_data,indices_arr_train,1)
    print (train_data.shape)
    validation_data=np.delete(validation_data,indices_arr_train,1)
    print (test_data.shape)
    test_data=np.delete(test_data,indices_arr_train,1)
    print (test_data.shape)
    
    
    
    train_data = train_data/255
    validation_data = validation_data/255
    
    
    
    #print 'train_data is : ' +  str(train_data.shape)
    #print 'validation_data is : ' +  str(validation_data.shape)
    #print 'train label size is' + str(train_label.shape)
    #print 'validaition label size is' + str(validation_label.shape)
    #print 'test label size is' + str(test_label.shape)
    
    
   # print train_data[-0]
    
    #for row in validation_label:
      #  print row
        
    #for row in train_label:    
     #   print row
    
    #print train_data[0].shape

                    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
        


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    obj_val = 0
    
    #Your code here
    #
    #
    #
    #
    #
    #print 'iteration start---------------------'

    n_iterations = 50000

    #print 'shape of w1 is', str(w1.shape)
    #print 'shape of w2 is', str(w2.shape)
    #print 'shape of train_data[0] is', train_data[0].shape
    
    
    a = np.array([])
    b = np.array([])
    z = np.array([])
    o = np.ndarray([])
    
    #print '\n---------------------'
    #feed forward from input to hidden layer
    #making bias vector, put in temp, multiply with w1 to get a
    bias1 = np.ndarray([50000,1])
    bias1.fill('1')
    #print 'bias1 shape is', bias1.shape

    temp1 = np.hstack((training_data, bias1))
    #print 'temp1 is', temp1.shape
    #print 'w1 transp is', np.transpose(w1).shape
    
    a = np.dot(temp1, np.transpose(w1))
    z = sigmoid(a)
    print 'shape z is', z.shape

    print '\n---------------------'
    #feed forward from hidden to output layer
    #making bias vector, put in temp, multiply with w2 to get b
    bias2 = np.ndarray([50000,1])
    bias2.fill('1')
    #print 'bias2 shape is', bias2.shape

    temp2 = np.hstack((z, bias2))
    print 'temp2 is', temp2.shape
    #print 'w2 transp is', np.transpose(w2).shape
    
    b = np.dot(temp2, np.transpose(w2))
    o = sigmoid(b)
    print 'shape o is', o.shape
    #print ' o is ', o

    print '\n---------------------'
    #calculate objective function
    #print np.log(0.58372467)
    J = (sum(sum(training_label*np.log(o) + ((1-training_label) * np.log(1-o)))))/(-n_iterations)
    print 'J=', J, ' nad its shape is ', J.shape



    #calculate error gradient
    deltaJO = np.ndarray([])

    deltaL = o-training_label
    #deltaL[np.newaxis,:]
    #+z[np.newaxis,:]
    print 'deltaL', deltaL.shape
    
    #deltaJO = np.dot(deltaL[0][:,np.newaxis], z[0][:,np.newaxis].T)
    #for i in range(1,n_iterations):
     #   deltaJO = np.dot(deltaL[i][:,np.newaxis], z[i][:,np.newaxis].T)

        
    #eq 9
    deltaJO = np.dot(np.transpose(deltaL), temp2)
    #
    print 'deltaJO shape', deltaJO.shape

    
    print '\n---------------------'
    #derivative of error function wrt weight from input to hidden layer ... eq 10-12    
    deltaJH = np.ndarray([])
    
    #w2_del = np.delete(w2, (-1), axis=1)
    #print 'w2 del shape', w2_del.shape
    
    #zt = np.transpose((1-temp2))
    de = (1-temp2)*temp2
    #print 'zt shape', zt.shape
    #print 'de shape', de.shape
    abc = np.dot(deltaL, w2)
    #print 'abc shape', abc.shape
    #eq12
    #deltaJH = np.dot(de , (np.dot(np.transpose(abc), temp1)))
    deltaJH = np.dot(np.transpose(de*abc),temp1)
    print 'deltaJH shape', deltaJH.shape
    #print deltaJH[0]
    
    
    #regularization
    #eq15
    weights_sum = sum(sum(w1*w1)) + sum(sum(w2*w2))
    #print 'weights_sum ',weights_sum
    Jbar = J + (lambdaval / (2*n_iterations)) * weights_sum
    #print 'Jbar', Jbar
        
    #eq16
    deltaJObar = (deltaJO + lambdaval*w2) / (n_iterations)
    #print deltaJObar.shape
    
    #eq17
    deltaJHbar = (np.delete(deltaJH, (-1), axis = 0) + lambdaval*w1) / (n_iterations)
    #print deltaJHbar.shape
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_val = Jbar
    grad_w1 = deltaJHbar
    grad_w2 = deltaJObar
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #print 'obj_grad ' , obj_grad
    print 'obj_val ' , obj_val
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
       
    bias1 = np.ndarray([data.shape[0],1])
    bias1.fill('1')
    
    temp1 = np.hstack((data, bias1))
    a = np.dot(temp1, np.transpose(w1))
    z = sigmoid(a)

    #feed forward from hidden to output layer
    #making bias vector, put in temp, multiply with w2 to get b
    bias2 = np.ndarray([data.shape[0],1])
    bias2.fill('1')

    temp2 = np.hstack((z, bias2))
    
    b = np.dot(temp2, np.transpose(w2))
    o = sigmoid(b)
    
    labels = np.argmax(o, axis=1)
    #print 'shape of labels', labels.shape
    return labels   



"""**************Neural Network Script Starts here********************************"""

start = timeit.default_timer()

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

np.seterr (over = 'ignore')

#  Train Neural Networkpredicted_label

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 80;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.4;

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)
print 'Predicted label shape :' , predicted_label.shape

#find the accuracy on Training Dataset
train_label = np.argmax(train_label, axis=1)
print train_label.shape
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#find the accuracy on Validation Dataset
predicted_label = nnPredict(w1,w2,validation_data)
validation_label = np.argmax(validation_label, axis=1)
print validation_label.shape
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#find the accuracy on Validation Dataset
predicted_label = nnPredict(w1,w2,test_data)
test_label = np.argmax(test_label, axis=1)
print test_label.shape
print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

end = timeit.default_timer()
print '\nn_input=',n_input
print 'n_hidden=',n_hidden
print 'lambdaval=', lambdaval
print ('Runtime :' + str(end - start))

##Data dump
print ('Writing to pickle file')
pickle.dump( [n_input, n_hidden, w1, w2, lambdaval], open( "params.pickle", "wb" ))
print ('Writing to pickle file done')



#with open('params.pickle', 'w') as g:
#    pickle.dump([n_input, n_hidden, w1, w2, lambdaval], g, protocol=pickle.HIGHEST_PROTOCOL)
