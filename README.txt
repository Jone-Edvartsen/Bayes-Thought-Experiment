Bayesian Thought Experiment


Packages Required:

numpy
matplotlib.pyplot
matplotlib.widgets
random
functools


Description:

This project visualizes a Bayesian and frequentist inference engine in action, with interactive buttons.

To run the program, execute: python BayesThoughtExperiment.py


General Overview:

This function addresses Bayes' thought experiment, demonstrating how incorporating prior information (prior belief) along with new data can improve one's beliefs and affect the resulting probabilities.


Thought Experiment:

Imagine you have an assistant who randomly drops a ball on a pool table while you are facing the other way. Initially, you believe the ball could be anywhere with equal probability. Your assistant then drops additional balls and tells you if they landed north, south, east, or west of the original ball. This additional information, combined with the prior belief, can be used to more accurately infer the position of the first ball. By dropping more balls, the process can be repeated, where previous improved beliefs become the prior beliefs when a new ball is dropped.


Technical Overview:

Solving bayes' thought experiment while visualizing the Bayesian updating of beliefs (and frequentist inference) in a visually pleasing manner.
Comparing the accuracy of Bayesian updating with a frequentist inference method through a Monte Carlo simulation for different amounts of additional data (additional balls being dropped).


The function therefore has two main parts:

1. Inference Problem and Visualization:

- Solving the inference problem for any number of additional balls being dropped.
- Visualizing how probabilities (representing the hypothesis that the first ball is in a given location) change with each additional ball.
Using a square defined by (x,y)∈[0,1]×[0,1] where balls occupy discrete positions with one significant digit for X and Y coordinates (e.g., 0.0, 0.1, 0.2, ..., 1.0).

2. Monte Carlo Simulation:

- Comparing the accuracy between the Bayesian and frequentist approaches as the number of additional balls increases.
- For the Bayesian approach, accuracy is computed by the Euclidean distance between the actual position of the first ball and the grid with the highest probability.
- For the frequentist approach, the distance is calculated between the actual position of the first ball and the closest possible discrete allowable position inferred from the ratio calculations.
- Running 1000 iterations for each amount of additional balls in the set S={1,2,3,...,200} with random placement of the first ball and additional balls to compute the average accuracy of both methods of inference.


Parameters in bayesian_vs_bernoulli (function doing the computation):

HowManyNewBalls (default=200):  Decides how many additional balls to infer and visualize for the first part of the function.

MC_simulations (default=1000):  Decides the number of random iterations for the second plot when determining the difference in accuracy between the Bayesian and frequentist methods.

HowManyGridPositionsInEachDirection (default=11): Determines the size of the pool table (and therefore the number of discrete possible locations balls can occupy). 11 means an 11x11 grid.


Theoretical Overview:

This thought experiment helped Bayes deduce the formula known as Bayes' theorem:

P(Hypothesis∣Data)= P(Data∣Hypothesis)×P(Hypothesis)/P(Data)
​
The formula takes in new information P(Data∣Hypothesis) and prior belief P(Hypothesis), normalizing by dividing by P(Data) to obtain an improved belief P(Hypothesis|Data).

In the thought experiment:

P(Data|Hypothesis): Probability of obtaining the specific information from an additional ball, given the hypothesis (specific grid location of the first ball).

P(Hypothesis): Prior information or initial belief, starting as an equal probability for each grid point.

P(Data): Normalization constant ensuring that probabilities of competing hypotheses sum to 1.

The improved belief is proportional to the information from the new data times the prior belief: P(Hypothesis∣Data)∝P(Data∣Hypothesis)×P(Hypothesis)

Since the first ball must be in one of the grid locations, the resulting probabilities are normalized so their sum equals 1.
