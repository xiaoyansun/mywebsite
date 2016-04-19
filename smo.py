#-------------------------------------------------------------------------------
# Name:        smo
# Purpose:
#
# Author:      Shi
#
# Created:     27/03/2016
# Copyright:   (c) Administrator 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import random
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
# to put kernels in a matrixs
import timeit
MIN_SUPPORT_VECTOR_MULTIPLIER=0.01

class smo():
    def __init__(self, C, tol, max_passes, X, Y, kernel):
        self.n_samples, self.n_features = X.shape
        self.alpha=np.zeros(self.n_samples)  # n_samples*1
        self.b=0
        self.tol=tol
        self.max_passes=max_passes
        self.C=C
        self.X=X
        self.Y=Y
        self.kernel=kernel
        #start = timeit.timeit()
        #print "startkernel: ", start
        self.K=self._getKernelMatrix()
        #end = timeit.timeit()
        #print "endkernel: ", end
        self.E=-self.Y

    def _getKernelMatrix(self):
        n_samples, n_features = self.X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(self.X):
            for j, x_j in enumerate(self.X):
                K[i, j] = self.kernel(x_i, x_j)
        return K

    def _updateE(self,a1, a2,b, i, j):
        #print i,j, a1, a2, b
        self.E=self.E+a1*self.Y[i]*self.K[i,:]+a2*self.Y[j]*self.K[j,:]+b
        #print self.K[i,:], self.K[j,:]

    def _getJ(self,e_i):
        if(e_i>=0):
            i=self.E>=0
            j=random.randint(0, len(i)-1)
            return i[j]
        else:
            i=self.E<0
            j=random.randint(0, len(i)-1)
            return i[j]

    def opt(self):
        #start = timeit.timeit()
        #print "start opt: ", start
        init_passes=0
        while(init_passes<self.max_passes):
            num_changed_alphas=0
            for i in range(0, self.n_samples-1):
                #f=self._f(i)
                #e_i=f-self.Y[i]
                e_i=self.E[i]
                if((self.Y[i]*e_i<-self.tol and self.alpha[i]<self.C)or (self.Y[i]*e_i>self.tol and self.alpha[i>0])):
                    #j=random.randint(0, self.n_samples-1)  #endpoints included
                    j=self._getJ(e_i)
                    while(j==i):
                        j=random.randint(0, self.n_samples-1)
                        #j=self._getJ(e_i)
                    #e_j=self._f(j)-self.Y[j]
                    e_j=self.E[j]
                    alpha_i_old=self.alpha[i]
                    alpha_j_old=self.alpha[j]
                    if(self.Y[i]!=self.Y[j]):
                        L=max(0,self.alpha[j]-self.alpha[i])
                        H=min(self.C,self.C+self.alpha[j]-self.alpha[i])
                    if(self.Y[i]==self.Y[j]):
                        L=max(0,self.alpha[i]+self.alpha[j]-self.C)
                        H=min(self.C,self.alpha[i]+self.alpha[j])

                    if(L==H):
                        continue

                    eta=2*self.K[i][j]-self.K[i][i]-self.K[j][i]
                    if(eta>=0):
                        continue
                    self.alpha[j]=self.alpha[j]-(self.Y[j]*(e_i-e_j))/eta
                    if(self.alpha[j]>H):
                        self.alpha[j]=H
                    elif(self.alpha[j]<L):
                        self.alpha[j]=L

                    a2=self.alpha[j]-alpha_j_old
                    if(abs(a2)<0.000001):
                        continue
                    self.alpha[i]=self.alpha[i]+self.Y[i]*self.Y[j]*(-a2)

                    a1=self.alpha[i]-alpha_i_old

                    b_1=self.b-e_i-self.Y[i]*(a1)*self.K[i][i]-self.Y[j]*(a2)*self.K[i][j]
                    b_2=self.b-e_j-self.Y[i]*(a1)*self.K[i][j]-self.Y[j]*(a2)*self.K[j][j]
                    b_old=self.b
                    if(self.alpha[i]>0 and self.alpha[i]<self.C):
                        self.b=b_1
                    elif(self.alpha[j]>0 and self.alpha[j]<self.C):
                        self.b=b_2
                    else:
                        self.b=(b_1+b_2)/2
                    num_changed_alphas+=1

                    #print i,j
                    #print e_i, e_j
                    #print self.alpha
                    #print self.E
                    self._updateE(a1,a2,self.b-b_old,i,j)
                    #print self.E.shape

                    #print self.E
                    #print self.alpha

            if(num_changed_alphas==0):
                init_passes+=1
            else:
                init_passes=0
        #end = timeit.timeit()
        #print "end opt: ", end
        return [self.alpha, self.b]

    def saveM(self, str):

        support_vector_indices = \
            self.alpha > MIN_SUPPORT_VECTOR_MULTIPLIER

        np.savez('multipliers/'+str, bias=self.b, support_multipliers=self.alpha[support_vector_indices],
            support_vectors=self.X[support_vector_indices], support_vector_labels=self.Y[support_vector_indices])

    def _f(self,j):
        sum=0
        for i in range(0,self.n_samples-1):
            sum+=self.alpha[i]*self.Y[i]*self.K[i,j]
        return sum+self.b

def main():
    import kernel
    num_samples=20
    num_features=2
    samples = np.random.normal(size=num_samples * num_features).reshape(num_samples, num_features)
    labels = 2 * (samples.sum(axis=1) > 0) - 1.0
    #samples=np.array([[2,1],[2,2],[0,1],[0,2],[0,4],[5,1],[2,3]])
    #labels=np.array([1,1,-1,-1,-1,1,1])
    s = smo(20,0.0001,4,samples,labels,kernel.Kernel.linear())
    res=s.opt()
    s.saveM('first')

    #print samples
    print res
    # plt.subplot(2, 1, 1)
    # plt.scatter(samples[:,0].ravel(), samples[:,1].ravel(), c=labels, alpha=0.5)
    #
    # plt.subplot(2, 1, 2)
    # plt.scatter(samples[:,0].ravel(), samples[:,1].ravel(), c=res[0], alpha=0.5)
    # plt.show()

if __name__ == '__main__':
    main()
