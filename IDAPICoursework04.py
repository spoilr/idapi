#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
from copy import deepcopy
import math

#
# Coursework 4 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here
    mean = numpy.mean(realData, axis=0)
    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here
    mean = Mean(theData)
    for i in range(noVariables):
        for j in range(noVariables):
            pairs = zip(realData[:,i], realData[:,j])
            covar[i][j] = sum(map(lambda (x,y):(x-mean[i])*(y-mean[j]), pairs)) / (len(realData) - 1) 
    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    # Coursework 4 task 3 begins here
    for j in range(len(theBasis)):
        filename = "PrincipalComponent" + str(j) + ".jpg"
        SaveEigenface(theBasis[j], filename)
    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here
    image = array(ReadOneImage(theFaceImage))
    magnitudes = dot(subtract(image, theMean), theBasis.transpose())
    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    # Coursework 4 task 5 begins here
    base = aMean
    SaveEigenface(base, "PartialReconstruction0.jpg")
    for j in range(len(componentMags)):
        base = base + dot(aBasis[j], componentMags[j])
        filename = "PartialReconstruction" + str(j+1) + ".jpg"
        SaveEigenface(base, filename)
    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 6 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes
    
    # UUt * evec = evec * eval 
    means = Mean(theData)
    U = subtract(theData, means)
    Ut = U.transpose()
    UUt = dot(U, Ut)
    eigenvalues, eigenvectors = linalg.eig(UUt)

    # UtU * evec = evec * eval where evec = Ut*(prev evec)
    eigenvectors = dot(Ut, eigenvectors)

    # normalize - divide eigenvector by its length
    for i in range(eigenvectors.shape[1]):
        length = linalg.norm(eigenvectors[:,i])
        eigenvectors[:,i] = eigenvectors[:,i] / length

    z = zip(eigenvalues, eigenvectors.transpose())
    z.sort(key = lambda x: x[0], reverse=True)
    orthoPhi = zip(*z)[1]

    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 4
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("IDAPIResults04.txt","1 - Coursework Four Results by oc511")
AppendString("IDAPIResults04.txt","2 - The mean vector Hepatitis C data set")
means = Mean(theData)
AppendList("IDAPIResults04.txt", means)
AppendString("IDAPIResults04.txt","3 - The covariance matrix of the Hepatitis C data set")
covar = Covariance(theData)
AppendArray("IDAPIResults04.txt", covar)
eigenfaceBasis = ReadEigenfaceBasis()
CreateEigenfaceFiles(eigenfaceBasis)
AppendString("IDAPIResults04.txt","4 - The component magnitudes for image “c.pgm” in the principal component basis used in task 4")
meanFace = array(ReadOneImage("MeanImage.jpg"))
projectedFace = ProjectFace(eigenfaceBasis, meanFace, "c.pgm")
AppendList("IDAPIResults04.txt", projectedFace)
CreatePartialReconstructions(eigenfaceBasis, meanFace, projectedFace)

# this will overwrite the previous images -- comment out to get previous results
imageData = array(ReadImages())
basis = PrincipalComponents(imageData)
means = Mean(imageData)
CreateEigenfaceFiles(basis)
projectedFace = ProjectFace(basis, means, "c.pgm")
CreatePartialReconstructions(basis, means, projectedFace)