# IRL algorith originally developed for the cart pole problem, modified to run on the toy car obstacle avoidance problem for testing
import numpy as np
import logging
import scipy
from playing import play #get the RL Test agent, gives out feature expectations after 2000 frames
from nn import neural_net #construct the nn and send to playing
from cvxopt import matrix
from cvxopt import solvers
from flat_game import carmunk
from learning import IRL_helper

NUM_SENSORS = 8


class irlAgent:
    def __init__(self): #initial constructor sorta function
        self.randomPolicy = [ 7.74363107 , 4.83296402 , 6.1289194  , 0.39292849 , 2.0488831  , 0.65611318 , 6.90207523 , 2.46475348]
        #self.expertPolicy = [  7.53667094e+00 ,  4.63506998e+00  , 7.44218366e+00  , 3.18175577e-01 ,  8.33987661e+00 ,  1.37107443e-08 ,  1.34194780e+00 ,  0.00000000e+00]#going all yellow
        #self.expertPolicy = [  7.91006787e+00  , 5.37453435e-01 ,  5.23635403e+00  , 2.86523487e+00 ,  3.31200074e+00  , 3.64787240e-06  , 3.82276074e+00  , 1.02196236e-17] # out and clock
        self.expertPolicy = [  5.22101668e+00  , 5.69809021e+00  , 7.79845852e+00  , 4.84405866e-01   ,    2.08859583e-04  , 9.22152114e+00  , 2.93864139e-01 ,  4.84985047e-17] #going all brown 0.9
        self.epsilon = 1.0
        self.randomT = np.linalg.norm(np.asarray(self.expertPolicy)-np.asarray(self.randomPolicy))
        self.policiesFE = {self.randomT:self.randomPolicy}
        print ("Expert - Random at the Start (t) :: " , self.randomT)
        self.currentT = self.randomT
        self.minimumT = self.randomT

    def getRLAgentFE(self, W): #get the feature expectations of a new poliicy using RL agent
        IRL_helper(W) # train the agent and save the model 
        saved_model = 'saved-models_brown/164-150-100-50000-25000.h5' # use the saved model to get the feature expectaitons
        model = neural_net(NUM_SENSORS, [164, 150], saved_model)
        return  play(model, W)#return feature expectations
    
    def policyListUpdater(self, W):  #add the policyFE list and differences
        tempFE = self.getRLAgentFE(W)
        hyperDistance = np.linalg.norm(np.asarray(self.expertPolicy)-np.asarray(tempFE))
        self.policiesFE[hyperDistance] = tempFE
        return hyperDistance
        
    def optimalWeightFinder(self):
        while True:
            if self.currentT <= self.minimumT:
                W = self.optimization() # update only upon finding a closer point
                print ("Weight Update ::", W )
            else:
                print ("current T is higher, learn again ")

            print ("weights ::", W )
            print ("the distances  ::", self.policiesFE.keys())
            self.currentT = self.policyListUpdater(W)
            self.minimumT = min(self.policiesFE)
            print ("Current distance (t) is:: ", self.currentT )
            if self.currentT < self.epsilon:
                break
        return W
    
    def optimization(self):
        m = len(self.expertPolicy)
        P = matrix(2.0*np.eye(m), tc='d') # min ||w||
        q = matrix(np.zeros(m), tc='d')
        #G = matrix((np.matrix(self.expertPolicy) - np.matrix(self.randomPolicy)), tc='d')
        policyList = [self.expertPolicy]
        h_list = [1]
        for i in self.policiesFE.keys():
            policyList.append(self.policiesFE[i])
            h_list.append(1)
        policyMat = np.matrix(policyList)
        policyMat[0] = -1*policyMat[0]
        G = matrix(policyMat, tc='d')
        h = matrix(-np.array(h_list), tc='d')
        sol = solvers.qp(P,q,G,h)

        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights = weights/norm
        return weights
                
            
            
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #rlEpisodes = 200
    #rlMaxSteps = 250
    #W = [-0.9, -0.9, -0.9, -0.9, 1]
    #env = gym.make('CartPole-v0')
    irlearner = irlAgent()
    #print irlearner.policiesFE
    #irlearner.policyListUpdater(W)
    #print irlearner.rlAgentFeatureExpecs(W)
    #print irlearner.expertFeatureExpecs()
    print (irlearner.optimalWeightFinder())
    #print irlearner.optimization(20)
    #np.squeeze(np.asarray(M))


