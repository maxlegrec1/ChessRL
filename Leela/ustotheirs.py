import numpy as np
import tensorflow as tf

def ustotheirs(X):
    #X shape : (B,8,8,102)
    X = tf.transpose(X,perm=[0,3,1,2])
    blank = tf.zeros((X.shape[0],1,8,8))
    ones = tf.ones((X.shape[0],1,8,8))
    list=[]
    for i in range(8):
        list.append(X[:,12*(7-i):12*(7-i+1),:,:])
        list.append(blank)
    list.append(X[:,-5:-4,:,:])#queensideours
    list.append(X[:,-6:-5,:,:])
    list.append(X[:,-3:-2,:,:])
    list.append(X[:,-4:-3,:,:])
    list.append(1-tf.expand_dims(X[:,-1,:,:],axis=1))
    list.append(blank)
    list.append(blank)
    list.append(ones)
    new_X=tf.concat(list,axis=1)

    return new_X




if __name__== "__main__":
    ours = tf.zeros((2,8,8,102))
    theirs = ustotheirs(ours)
    print(theirs.shape)