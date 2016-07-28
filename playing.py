"""
Once a model is learned, use this to play it. that is run/exploit a policy to get the feature expectations of the policy
"""

from flat_game import carmunk
import numpy as np
from nn import neural_net
import sys
import time

NUM_STATES = 8
GAMMA = 0.9


def play(model, weights):

    car_distance = 0
    game_state = carmunk.GameState(weights)

    _, state, __ = game_state.frame_step((2))

    featureExpectations = np.zeros(len(weights))

    # Move.
    #time.sleep(15)
    while True:
        car_distance += 1

        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))
        #print ("Action ", action)

        # Take action.
        immediateReward , state, readings = game_state.frame_step(action)
        #print ("immeditate reward:: ", immediateReward)
        #print ("readings :: ", readings)
        #start recording feature expectations only after 100 frames
        if car_distance > 100:
            featureExpectations += (GAMMA**(car_distance-101))*np.array(readings)
        #print ("Feature Expectations :: ", featureExpectations)
        # Tell us something.
        if car_distance % 2000 == 0:
            print("Current distance: %d frames." % car_distance)
            break


    return featureExpectations

if __name__ == "__main__": # ignore
    BEHAVIOR = sys.argv[1]
    ITERATION = sys.argv[2]
    FRAME = sys.argv[3]
    saved_model = 'saved-models_'+BEHAVIOR+'/evaluatedPolicies/'+str(ITERATION)+'-164-150-100-50000-'+str(FRAME)+'.h5'
    weights = [-0.79380502 , 0.00704546 , 0.50866139 , 0.29466834, -0.07636144 , 0.09153848 ,-0.02632325 ,-0.09672041]
    model = neural_net(NUM_STATES, [164, 150], saved_model)
    print (play(model, weights))
