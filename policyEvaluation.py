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
   
    weights = [[ 0.25113026  ,0.12665933 , 0.2337365 , -0.24902441 ,-0.41625774 ,-0.36763118, -0.45510049, -0.53731318],
    [ 0.27274123  ,0.13370425 , 0.24796427 ,-0.27907016 ,-0.44647981 ,-0.14933032 ,-0.48362556 ,-0.55931343],
    [ 0.32691606  ,0.18549009 , 0.27767445 ,-0.2634921  ,-0.53513236 ,-0.17592619, -0.57946415 ,-0.24309005],
    [ 0.29869779 ,-0.10281008  ,0.33087001 ,-0.60425918 ,-0.41447044 ,-0.19804659 ,-0.38206953, -0.26195472],
    [ 0.17593456 , 0.17575453  ,0.11978925 ,-0.77919219 ,-0.29653833, -0.19485005 ,-0.14562471 ,-0.41177587],
    [ 0.60655281 ,-0.24462372 ,-0.48915146 ,-0.37983326  ,0.13192824 ,-0.0807099 ,-0.3652799 , -0.17709299],
    [ 0.53579818  ,0.06821153 ,-0.66356037 ,-0.38756875  ,0.16607826 ,-0.07549711, -0.27436563 ,-0.09585038],
    [ 0.68721122 ,-0.25996475 ,-0.44837155  ,0.09193914 , 0.01417313 ,-0.14971882 ,-0.11218169 ,-0.46418518],
    [-0.08805555 ,-0.06245599  ,0.09146864 ,-0.01147858  ,0.66908548 ,-0.07713598 ,-0.66502319, -0.28976889]] #manually enter the weights obtained
    i = 1
    for W in weights:
    	IRL_sorter(W, i)
    	i += 1