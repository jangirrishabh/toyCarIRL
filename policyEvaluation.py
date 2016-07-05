"To manually evaluate all the policies obtained after convergence of the IRL step"

import numpy as np
import logging
import scipy
from playing import play #get the RL Test agent, gives out feature expectations after 2000 frames
from nn import neural_net #construct the nn and send to playing
from cvxopt import matrix
from cvxopt import solvers
from flat_game import carmunk
from learning import IRL_sorter

NUM_SENSORS = 8


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
   
    weights = [[ 0.20589866, -0.67237298 ,-0.29958379 ,-0.09453022, -0.19244134 , 0.59977423 ,-0.10034259, -0.00157184],
[ 0.2657772 , -0.75727577, -0.18523371, -0.13018025 ,-0.17342122 , 0.52233142 ,-0.04159359 ,-0.00154845],
[ -4.42675600e-01 , -1.63303871e-01  , 3.60230970e-01  ,-6.28545248e-01 , -1.78029459e-01 ,  2.79458569e-01  ,-3.77813515e-01  ,-2.14724924e-04],
[-0.01708273, -0.10464621, -0.0759424 , -0.08959176 ,-0.21147643  ,0.75502266 ,-0.56976002, -0.18856151],
[-0.26276059  ,0.03635897 , 0.0931208  , 0.00469231 ,-0.18295949  ,0.69874527 ,-0.59225935 ,-0.22011618]] #manually enter the weights obtained
    i = 1
    for W in weights:
    	IRL_sorter(W, i)
    	i += 1