#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
from copy import deepcopy
import math
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    rootValues = theData[:,root]
    occurences = numpy.bincount(rootValues) # The states are numbered consecutively from 0 upwards
    occTotal = sum(occurences)
    prior = array(occurences, float) / occTotal
# end of Coursework 1 task 1
    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserted here
    CValues = theData[:,varC]
    PValues = theData[:,varP]
    CStates = noStates[varC]
    PStates = noStates[varP]
    COcc = numpy.bincount(PValues)
    for i in range(CStates):
        for j in range(PStates):
            if COcc[j] > 0:
                joint = len([k for k in range(len(theData)) if CValues[k] == i and PValues[k] == j])
                cPT[i][j] = float(joint) / COcc[j]
# end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here 
    rowValues = theData[:,varRow]
    colValues = theData[:,varCol]
    dataPoints = len(theData)
    rowStates = noStates[varRow]
    colStates = noStates[varCol]
    for i in range(rowStates):
        for j in range(colStates):
            joint = len([x for x in range(len(theData)) if rowValues[x] == i and colValues[x] == j])
            jPT[i][j] = float(joint) / dataPoints
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here 
    cols = aJPT.shape[1]
    for i in range(cols):
        currentCol = aJPT[:, i]
        alpha = float(1) / sum(currentCol)
        aJPT[:, i] = alpha * currentCol
# coursework 1 taks 4 ends here
    return aJPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    evidence = 1
    for i in range(len(theQuery)):
        evidence *= naiveBayes[i+1][theQuery[i],:]
    rootPdf = transpose(naiveBayes[0]) * evidence
    alpha = float(1) / sum(rootPdf)    
    rootPdf = alpha * rootPdf
# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here
    numRows = jP.shape[0]
    numCols = jP.shape[1]
    probForEachRow = [sum(jP[i,:]) for i in range(numRows)]
    probForEachCol = [sum(jP[:,i]) for i in range(numCols)]
    for i in range(numRows):
        for j in range(numCols):
            if (jP[i][j] != 0.0):
                mi += jP[i][j] * math.log(jP[i][j] / (probForEachRow[i] * probForEachCol[j]), 2)
# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    for i in range(noVariables):
        for j in range(noVariables):
            jpt = JPT(theData, i, j, noStates)        
            MIMatrix[i][j] = MutualInformation(jpt)
# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    numRows = depMatrix.shape[0]
    numCols = depMatrix.shape[1]
    for i in range(numRows):
        for j in range(i+1, numCols):
            if i != j:
                depList.append([depMatrix[i][j], i, j])
    depList2 = sorted(depList, key=lambda x: x[0], reverse=True)        
# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def isPath(adjList, node1, node2, visited=[]):
    visited = visited + [node1]
    if node1 == node2:
        return True
    for node in adjList[node1]:
        if node not in visited:
            path = isPath(adjList, node, node2, visited)
            if path:
                return path
    return False    

def adjacencyList(noVariables):
    adjList = dict()
    for i in range(noVariables):
        adjList[i] = []
    return adjList     

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    adjList = adjacencyList(noVariables)
    for x in depList:
        node1 = x[1]
        node2 = x[2]
        if not isPath(adjList, node1, node2):
            adjList[node1].append(node2)
            adjList[node2].append(node1)
            spanningTree.append([x[0], node1, node2])
    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
    parentsCombinationsOcc = zeros([noStates[parent1],noStates[parent2]], float )
    for d in theData:
        cPT[d[child]][d[parent1]][d[parent2]] += 1
        parentsCombinationsOcc[d[parent1]][d[parent2]] += 1

    for i in range(parentsCombinationsOcc.shape[0]):
        for j in range(parentsCombinationsOcc.shape[1]):
            occ = parentsCombinationsOcc[i][j]
            if occ > 0:
                for k in range(noStates[child]):
                    cPT[k][i][j] /= occ
            else:
                # if a parent combination never occurs, then children cases have equal prob
                cPT[k][i][j] /= noStates[child]
# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here

def HepatitisCNetwork(theData, noStates):
    arcList = [[0], [1], [2,0], [3,4], [4,1], [5,4], [6,1], [7,0,1], [8,7]]
    cptList = []
    cptList.append(Prior(theData, 0, noStates))
    cptList.append(Prior(theData, 1, noStates))
    cptList.append(CPT(theData, 2, 0, noStates))
    cptList.append(CPT(theData, 3, 4, noStates))
    cptList.append(CPT(theData, 4, 1, noStates))
    cptList.append(CPT(theData, 5, 4, noStates))
    cptList.append(CPT(theData, 6, 1, noStates))
    cptList.append(CPT_2(theData, 7, 0, 1, noStates))
    cptList.append(CPT(theData, 8, 7, noStates))
    return arcList, cptList
# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here
    for x in cptList:
        # prior probability with m probability values can be represented by m − 1 parameters
        parameters = x.shape[0] - 1
        # link matrix with n × m probability values can be represented by (n − 1) × m parameters (n is the nr states of child)
        for j in range(1, len(x.shape)):
            parameters *= x.shape[j]
        mdlSize += parameters

    mdlSize *= math.log(noDataPoints, 2)
    mdlSize /= 2
# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here
    for nodeArcs in arcList:
        currentNode = nodeArcs[0]
        currentCpt = cptList[currentNode]
        firstNode = dataPoint[currentNode]
        if len(nodeArcs) == 1:
            jP *= currentCpt[firstNode]
        else:    
            parents = nodeArcs[1:]
            if len(parents) == 1:
                jP *= currentCpt[firstNode][dataPoint[parents[0]]]
            else:
                jP *= currentCpt[firstNode][dataPoint[parents[0]]][dataPoint[parents[1]]]
# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here
    for dataPoint in theData:
        jp = JointProbability(dataPoint, arcList, cptList)
        mdlAccuracy += math.log(jp, 2)
# Coursework 3 task 5 ends here 
    return mdlAccuracy

def MDLScore(theData, arcList, cptList, noDataPoints, noStates):
    return MDLSize(arcList, cptList, noDataPoints, noStates) - MDLAccuracy(theData, arcList, cptList)

def BestNetworkScoreRemovingOneArc(theData, arcList, cptList, noDataPoints, noStates):
    nets = []
    arcsRemoved = []
    for index, nodeArcs in enumerate(arcList):
        if len(nodeArcs) > 1:
            parents = nodeArcs[1:]
            for p in parents:
                newParents = deepcopy(parents)
                newParents.remove(p)
                newArcs = deepcopy(arcList)
                newArcs[index] = [nodeArcs[0]] + newParents
                newCpts = cptListFromArcList(theData, newArcs, noStates)
            score = MDLScore(theData, newArcs, newCpts, noDataPoints, noStates)    
            nets.append((newArcs, score))    
            arcsRemoved.append(([nodeArcs[0], p], score))        
    return min(nets, key=lambda x: x[1])

def cptListFromArcList(theData, arcList, noStates):
    cptList = []
    for arcs in arcList:
        if len(arcs) == 1:
            cptList.append(Prior(theData, arcs[0], noStates))
        if len(arcs) == 2:
            cptList.append(CPT(theData, arcs[0], arcs[1], noStates))
        if len(arcs) == 3:    
            cptList.append(CPT_2(theData, arcs[0], arcs[1], arcs[2], noStates))
    return cptList

#
# End of coursework 3
#
# Coursework 4 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)


def createNetwork(theData, noVariables, noRoots, noStates):
    naiveBayes = []
    for i in range(noRoots):
        naiveBayes.append(Prior(theData, i, noStates))
    noNonRoots = noVariables - noRoots    
    for i in range(noNonRoots):
        var = i + noRoots
        naiveBayes.append(CPT(theData, var, 0, noStates))
    return array(naiveBayes)

#
# main program part for Coursework 3
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("IDAPIResults03.txt","1 - Coursework Three Results by oc511")

arcList, cptList = HepatitisCNetwork(theData, noStates)
mdlSize = MDLSize(arcList, cptList, noDataPoints, noStates)
AppendString("IDAPIResults03.txt","2 - The MDLSize of the network for Hepatitis C data set: %f" % mdlSize)
mdlAccuracy = MDLAccuracy(theData, arcList, cptList)
AppendString("IDAPIResults03.txt","3 - The MDLAccuracy of the network for Hepatitis C data set: %f" % mdlAccuracy)
mdlScore = MDLScore(theData, arcList, cptList, noDataPoints, noStates)
AppendString("IDAPIResults03.txt","4 - The MDLScore of the network for Hepatitis C data set: %f" % mdlScore)
bestNetwork, bestNetworkScore = BestNetworkScoreRemovingOneArc(theData, arcList, cptList, noDataPoints, noStates)
AppendString("IDAPIResults03.txt","5 - The score of the best network with one arc removed: %f" % bestNetworkScore)
AppendString("IDAPIResults03.txt","5 - The best network with one arc removed %s" % bestNetwork)
