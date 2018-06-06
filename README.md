# Using Inverse Reinforcement Learning to train a toy car in a 2D game to learn given Expert Behaviors

**Note: The RL algorithm and the simulation game used is the work of Matthew Harvey. Thank you Matt**

To know more please visit my blog https://jangirrishabh.github.io/2016/07/09/virtual-car-IRL/



# Apprenticeship learning using Inverse Reinforcement Learning

Reinforcement learning (RL) is is the very basic and most intuitive form of trial and error learning, it is the way by which most of the living organisms with some form of thinking capabilities learn. Often referred to as learning by exploration, it is the way by which a new born human baby learns to take its first steps, that is by taking random actions initially and then slowly figuring out the actions which lead to the forward walking motion.

Now the question that I kept asking myself is, what is the driving force for this kind of learning, what forces the agent to learn a particular behavior in the way it is doing it. Upon learning more about RL I came across the idea of rewards, basically the agent tries to choose its actions in such a way that the rewards that is gets from that particular behavior are maximized. Now to make the agent perform different behaviors, it is the reward structure that one must modify/exploit. But assume we only have the knowledge of the behavior of the expert with us, then how do we estimate the reward structure given a particular behavior in the environment? Well, this is the very problem of Inverse Reinforcement Learning (IRL), where given the optimal expert policy (actually assumed to be optimal), we wish to determine the underlying reward structure.

Again, this is not an Intro to Inverse Reinforcement Learning post, rather it is a tutorial on how to use/code Inverse reinforcement learning framework for your own problem, but IRL lies at the very core of it, and it is quintessential to know about it first. IRL has been extensively studied in the past and algorithms have been developed for it, please go through the papers Ng and Russell,2000, and Abbeel and Ng, 2004 for more information.

### Install Pygame

Install Pygame's dependencies with:

`sudo apt install mercurial libfreetype6-dev libsdl-dev libsdl-image1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libsdl-mixer1.2-dev libswscale-dev libjpeg-dev`

Then install Pygame itself:

`pip3 install hg+http://bitbucket.org/pygame/pygame`

### Install Pymunk

This is the physics engine used by the simulation. It just went through a pretty significant rewrite (v5) so you need to grab the older v4 version. v4 is written for Python 2 so there are a couple extra steps.

Go back to your home or downloads and get Pymunk 4:

`wget https://github.com/viblo/pymunk/archive/pymunk-4.0.0.tar.gz`

Unpack it:

`tar zxvf pymunk-4.0.0.tar.gz`

Update from Python 2 to 3:

`cd pymunk-pymukn-4.0.0/pymunk`

`2to3 -w *.py`

Install it:

`cd ..`
`python3 setup.py install`

Now go back to where you cloned `reinforcement-learning-car` and make sure everything worked with a quick `python3 learning.py`. If you see a screen come up with a little dot flying around the screen, you're ready to go!

## Training

First, you need to train a model. This will save weights to the `saved-models` folder. *You may need to create this folder before running*. You can train the model by running:

`python3 learning.py`

It can take anywhere from an hour to 36 hours to train a model, depending on the complexity of the network and the size of your sample. However, it will spit out weights every 25,000 frames, so you can move on to the next step in much less time.

## Playing

Edit the `playing.py` file to change the path name for the model you want to load. Sorry about this, I know it should be a command line argument.

Then, watch the car drive itself around the obstacles!

`python3 playing.py`

That's all there is to it.

# Resources and credits -

1. Thank you Matt Harvey for the game and the RL framework, basically for the father blog to this blog - Using RL to teach a virtual car to avoid obstacles
2. Andrew Ng and Stuart Russell, 2000 - Algorithms for Inverse Reinforcement Learning
3. Pieter Abbeel and Andrew Ng, 2004 - Apprenticeship Learning via Inverse Reinforcement Learning
