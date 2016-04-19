#-------------------------------------------------------------------------------
# Name:        trainer
# Purpose:
#
# Author:      Shi
#
# Created:     31/03/2016
# Copyright:   (c) Administrator 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import smo
#import matplotlib.pyplot as plt

MIN_SUPPORT_VECTOR_MULTIPLIER=0.01

class trainer(object):
    def __init__(self, kernel, c, id):
        self._kernel = kernel
        self._c = c
        self.id=id  #type string

    def train(self, X, y):
        """Given the training features X with labels y, returns a SVM
        predictor representing the trained SVM.
        """
        self._compute_multipliers(X,y)

    def _compute_multipliers(self, X, y):
        tol=0.001
        passes=4
        s=smo.smo(self._c, tol, passes, X, y, self._kernel)
        res=s.opt()
        #print res
        s.saveM(self.id)
        #plt.subplot(2, 1, 2)
        #plt.scatter(X[:,0].ravel(), X[:,1].ravel(), c=res[0], alpha=0.5)
        #plt.show()
