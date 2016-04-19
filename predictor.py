#-------------------------------------------------------------------------------
# Name:        predictor
# Purpose:
#
# Author:      Shi
#
# Created:     31/03/2016
# Copyright:   (c) Administrator 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import kernel

MIN_SUPPORT_VECTOR_MULTIPLIER=0.01
class predictor():

    def __init__(self, kernel, id):
        self._bias=np.load('multipliers/'+id+'.npz')['bias']

        self._support_multipliers = np.load('multipliers/'+id+'.npz')['support_multipliers']
        self._support_vectors = np.load('multipliers/'+id+'.npz')['support_vectors']
        self._support_vector_labels = np.load('multipliers/'+id+'.npz')['support_vector_labels']
        self._kernel=kernel

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._bias
        for z_i, x_i, y_i in zip(self._support_multipliers,
                                 self._support_vectors,
                                 self._support_vector_labels):

            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()