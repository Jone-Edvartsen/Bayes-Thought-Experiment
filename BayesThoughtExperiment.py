import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
from functools import partial


def bayesian_vs_bernoulli(HowManyNewBalls=200, MC_simulations=1000, HowManyGridPositionsInEachDirection=11):
    """ General Overview:
    
        This function addresses Bayes' thought experiment, which led to the realization
        that incorporating prior information (prior belief) along with new data
        can improve one's beliefs and affect the resulting probabilities.

        The thought experiment goes as follows:
        Imagine you have an assistant who randomly drops a ball on a pool table
        while you are facing the other way. Initially, you have no idea where the ball is on 
        the table, so you believe it could be anywhere with equal probability. However, your
        assistant then drops an additional ball and tells you if it landed to the north, 
        south, east, or west of the original ball. This additional information (the direction 
        relative to the first ball), along with the prior belief (that the first ball has an 
        equal probability of being anywhere on the table), can be used to more accurately 
        infer the position of the first ball (improved belief). By dropping more balls, 
        this process can be repeated, where the previous improved beliefs become the prior 
        beliefs when a new ball is dropped, and is iteratively used to infer a new improved
        belief.
        

        Technical Overview:
        
        The function have two main parts. Solving bayes thought experiment using bayesian updating
        of beliefs in a visually pleasing manner, and, comparing the accuracy of the bayesian 
        updating method with a frequentist inference method through a Monte Carlo simulation for
        different amounts of additional data (additional balls being dropped).

        In the first part, we will solve the inference problem for any number of additional balls
        being dropped and visualize how the probabilities (representing the hypothesis that the first
        ball is any given location) change with each additional ball. To simplify,we will use a square defined 
        by (x, y) ∈ [0, 1] x [0, 1], where the balls can only occupy discrete positions with one significant digit 
        for the X and Y coordinates (e.g., 0.0, 0.1, 0.2, ..., 1.0). Additional balls can also not obtain the exact 
        same position as the first ball, however, multiple additional balls are allowed to occupy the same position. 
        If an additional ball would occupy the same position as the first ball, one cannot get information on the 
        additional ball being either north/south and east/west of the first ball, which is the reason why we disallow 
        this. For additional balls, we allow these to occupy identical positions. Else, our square would fill up after 
        only 11 x 11 = 121 balls, and would hinder analysis beyond this amount of balls. Furthermore, the probabilities
        inferred from the Bayesian process will be dynamically visualized by coloring of grids, and with a white square
        representing the position of the first ball. In addition, the visualization will also show a frequentists approach, 
        where the X (west-east) and Y (south-north) location of the first ball is continuously inferred in a two step process. 
        Firstly, we calculate the X-position and Y-position according to the ratios:
        X-position = 1 - Number of balls to the east / Total number of balls dropped
        Y-position = 1 - Number of balls to the north / Total number of balls dropped.
        Then, since all balls can only occupy discrete locations, the closest possible discrete allowable position to the calculated
        X- and Y-position will be used as the final location inferred from the frequentist approach.
        
        In the second part, we will compute a Monte Carlo (MC) simulation comparing the accuracy between the bayesian approach
        and frequentist approach as the number of additional balls increases. For the bayesian approach, the accuracy is 
        computed by the euclidean distance between the actual position of the first ball and the grid with the highest probability
        obtained by the bayesian method, while for the frequentist approach, the distance is calculated between the actual position
        of the first ball and the closest possible discrete allowable position inferred from the ratios calculations. The 
        MC simulation works by running the inference code for a 1000 times, for each of the amounts of additional balls in the 
        set S = {1, 2, 3, ..., 200}, where each iteration assures random placement of first ball and random placement of additional balls. 
        The aggregated total of 1000 iterations for each case of additional balls in the set S is used to give accurate statistical
        information about the two methods.
        
        In general, the function takes the arguments:
        
        HowManyNewBalls (default=200), which decides how many additional balls to
        infer and visualize for the first part of the funtion.
        
        MC_simulations (default=1000), which decides the amount of random iterations for second plot when determining the difference in accuracy
        between bayesian and frequentist method.
        
        HowManyGridPositionsInEachDirection (default=11), which determines the size of the pool table (and therefore also the number of discrete possible
        locations balls can occupy). 11 means an 11x11 grid.
        
        
        Theoretical Overview:
         
        This thought experiment helped Bayes when deducing the formula known as Bayes theorem,
        which we will use when solving the thought experiment.
        
        Bayes theorem:  P(Hypothesis|Data) = P(Data|Hypothesis)*P(Hypothesis) / P(Data)
        
        The formula takes in new information, P(Data|Hypothesis), and prior belief, P(Hypothesis),
        normalizes this by dividing by P(Data), to obtain an improved belief, P(Hypothesis|Data).
        
        We can further relate the elements in Bayes theorem to the thought experiment we will solve:
        
        P(Data|Hypothesis) is the probability of the information (south-north and west-east of the first ball) 
        given by an additional ball being dropped (Data), given the Hypothesis, which is simply the location of 
        the first ball being in a specific grid. To clarify further, if H_1 is the hypothesis that the first ball 
        is in location (X=0, Y=0), P(Data|H_1) is the probability of obtaining the specific information from an 
        additional ball (which could be for example north and east of first ball), given that the first ball is 
        at location (0, 0). From this, it becomes obvious that each grid represents an independent hypothesis,
        that the first ball is in this specific location.
        
        P(Hypothesis) represents the prior information or initial belief. When starting out the experiment,
        each discrete grid point have an equal probability of housing the first ball, which means that P(Hypothesis) for
        each grid should be 1/121. In other words, the probability of the ball being in a specific grid is equal to 1/121.
        
        P(Data) normally functions as a normalization constant and is often given by the equation: 
        P(Data) = P(Data|Hypothesis) * P(Hypothesis) + (1-P(Data|Hypothesis)) * (1-P(Hypothesis)). This normalization 
        constant ensures that the probabilities of competing hypotheses sum to 1. However, since we are dealing with 
        specific grids, where the first ball must be located in one of the grids, we need to normalize the resulting 
        probabilities such that the sum of the probabilities of the first ball being located in any of the grids equals 
        one, instead of using the general normalization procedure above.
        
        P(Hypothesis|Data) then represents the improved belief (improved probability) that the first ball is in a 
        specific grid, given the data of an additional ball (e.g. north, east). When the next additional ball is dropped, 
        P(Hypothesis|Data) becomes the new prior (P(Hypothesis))
        
        To clarify the procedure, including the updating and the normalization.
        
        P(Hypothesis|Data) ∝ P(Data|Hypothesis)*P(Hypothesis) - At any specific grid, the improved belief is proportional 
        to the information from the new data times the prior belief.
        
        P(Hypothesis|Data) now represents the improved belief in every grid (likeli with differing probabilities in each grid), 
        and can be visualized as;
        
        [P(H_1|Data), P(H_2|Data), P(H_3|Data),
         P(H_4|Data), P(H_5|Data), P(H_6|Data),
         P(H_5|Data), P(H_8|Data), P(H_9|Data)] 
         
        for a 3x3 grid. Since the first ball must be in one of the locations, these resulting probabilities needs to add up to 1.
        This normalization is simply done by summing up the values of the resulting probabilities, and dividing the value in each
        specific grid by this sum.
        
        
        """
    
    # Choosing random coordinates for first ball
    def RandomBallCoordinates():
        randomCoordinates = [random.choice(randomArray), random.choice(randomArray)]
        return randomCoordinates[0], randomCoordinates[1]
    
    
    # A nested function is used, because a play button is added to the figure.
    # Every time the play button is clicked, the function run_single_inference runs again.
    def run_single_inference(HowManyNewBalls, plot=True, event=None):
        
        # This is false until the function is run once. It assures that after the first run, this function will generate a random location for the first ball itself
        nonlocal AlreadyClickedButton
        
        # Initilize an empty list which will store the eucledian distances between the position of the first ball
        StoreEuclideanDistances = []
        
        # Only activate when plotting function is used
        if plot:
            # Clear the previous plot since we want dynamic visualization
            ax_1.clear()
            fig.canvas.draw_idle()
        
        # Unly use the AlreadyClickedButton functionality when in plotting mode
        if plot: 
            # If this is true, it means the start button is clicked, and we need to generate new random positions for the first ball after this
            if AlreadyClickedButton is True:
                FirstBall_xy[0], FirstBall_xy[1] = RandomBallCoordinates()
        else:
            FirstBall_xy[0], FirstBall_xy[1] = RandomBallCoordinates()
            
        # List to store coordinates of additionall balls
        AdditionalBalls = []
        # Counter for the amount of additional balls
        AdditionalBallsCounter = 0
        
        # Add additional balls up to HowManyNewBalls, while assuring that none of the additional balls occupy the position of the first ball
        while AdditionalBallsCounter < HowManyNewBalls:
            # Random coordinates for additional ball
            x_corr, y_corr = RandomBallCoordinates()

            # If position is equal to first ball, continue the next iteration of loop (maybe not the most efficient way of handling this)
            if x_corr == FirstBall_xy[0] and y_corr == FirstBall_xy[1]:
                continue

            # Assign information to the additional balls dependening on their position relative to the first ball
            
            # A larger X-cor means east of first ball, and is assigned 1
            if x_corr > FirstBall_xy[0]:
                x_compare = 1
            # A smaller X-cor means west of first ball, and is assigned -1    
            elif x_corr < FirstBall_xy[0]:
                x_compare = -1
            # Similar X-cor as first ball is assigned 0    
            else:
                x_compare = 0

            # A larger Y-cor means north of first ball, and is assigned 1
            if y_corr > FirstBall_xy[1]:
                y_compare = 1
            # Smaller Y-cor means south of first ball, and is assigned -1    
            elif y_corr < FirstBall_xy[1]:
                y_compare = -1
            # Similar Y-cor as first ball is assigned 0     
            else:
                y_compare = 0
                
            # Increment counter    
            AdditionalBallsCounter += 1    

            # Append balls to list
            AdditionalBalls.append([x_compare, y_compare])

        # Convert list to array
        AdditionalBalls = np.array(AdditionalBalls)
        
        # Initial probability a long the X-dir and Y-dir, respectively.
        # There are 11 rows and 11 columns, forming the grid square.
        # There is initially a 1/11 probability of the first ball being in any of the rows, and 1/11 of the first ball being in any of the columns.
        # This is used since we treat X-dir and Y-dir separately
        initial_value = 1 / len(X[0])
        # Make it the shape of the grid
        Z = initial_value * np.ones(X.shape)

        # This is the interesting part. Here, we need to loop through all balls,
        # and update the probabilities correctly according to the bayesian and frequentist method.
        for allAdditionalBalls in range(len(AdditionalBalls)):
            
            # In first iteration, assign initial priors
            if allAdditionalBalls == 0:
                PriorX = Z.copy()
                PriorY = Z.copy()
            # For all other iterations, assign the improved belief from last iteration as the new prior    
            else:
                PriorX = P_hypothesis_given_data_X_normalized.copy()
                PriorY = P_hypothesis_given_data_Y_normalized.copy()
                
            # Initilize data probabilities for the X-dir and Y-dir as grids of correct shape with zeros
            DataProbability_X = np.zeros_like(Z)
            DataProbability_Y = np.zeros_like(Z)
            
            
            # This step requires some thinking. Data probability is really the data probability given
            # the hypothesis, or P(Data|Hypothesis). The conditional probability statement P(Data|Hypothesis) can be 
            # thought of as the probability of obtaining the data, given that a specific hypothesis is assumed correct. 
            # Given a certain new ball (data point), which tells us if the additional ball is either south(-1), similar(0), or north(1); and either east(-1), similar(0) or west(1),
            # how can we correctly calculate the probability of obtaining the specific data, given the hypothesis that the first ball is in a given location?
            #
            # Let's start by reiterating that the different hypotheses can be stated in this manner:
            # Hypothesis (x = 0):   The first ball is at x = 0
            # Hypothesis (x = 0.1): The first ball is at x = 0.1,
            # Hypothesis (x = 0.2): The first ball is at x = 0.2,
            # Hypothesis (x = 0.3): The first ball is at x = 0.3,
            # Hypothesis (x = 0.4): The first ball is at x = 0.4,
            # Hypothesis (x = 0.5): The first ball is at x = 0.5,
            # Hypothesis (x = 0.6): The first ball is at x = 0.6,
            # Hypothesis (x = 0.7): The first ball is at x = 0.7,
            # Hypothesis (x = 0.8): The first ball is at x = 0.8,
            # Hypothesis (x = 0.9): The first ball is at x = 0.9,
            # Hypothesis (x = 1):   The first ball is at x = 1, with a similar logical applying for the y direction.
            # 
            # Given that we assume hypothesis (x=0) to be correct (the first ball is at x = 0), P(Data)_X = west(1) is 10/11.
            # In words, if the first ball is at the westernmost location, additional balls can in 10/11 cases occupy a placement to the east of it, and in 1/11 cases occupy the same placement.
            # Similarly, for the hypothesis (x=0.1), there is 9/11 possible locations to the east, 1/11 location being similar, and 1/11 locations to the west of it.
            # The end case, for the hypothesis (x=1), there is no locations to the east of this, which would make P(Data=East|x=1) = 0.
            # Since additional balls occupy these locations with uniform probability, the number of cases is equivalent to the probability of the data (in this case east) given a specific x coordinate of the first ball.
            #
            # Using this logic, we can write out P(Data=East|Hypothesis X-dir) for the individual hypotheses along the X-dir:
            #
            # P(Data=East|x=0.0) = 10/11
            # P(Data=East|x=0.1) = 9/11
            # P(Data=East|x=0.2) = 8/11
            # P(Data=East|x=0.3) = 7/11
            # P(Data=East|x=0.4) = 6/11
            # P(Data=East|x=0.5) = 5/11
            # P(Data=East|x=0.6) = 4/11
            # P(Data=East|x=0.7) = 3/11
            # P(Data=East|x=0.8) = 2/11
            # P(Data=East|x=0.9) = 1/11
            # P(Data=East|x=1.0) = 0
            #
            # Since the probability a long the X-direction must sum to 1, these need to be normalized by dividing each probability by the total sum (10/11 + 9/11 + 8/11 + ... + 1/11 = 5), giving:
            #
            # P(Data=East|x=0.0) = 10/11 / 5 = 2/11 = 0.1818..
            # P(Data=East|x=0.1) = 9/11 / 5 = 9/55 = 0.1636..
            # P(Data=East|x=0.2) = 8/11 / 5 = 8/55 = 0.1454..
            # P(Data=East|x=0.3) = 7/11 / 5 = 7/55 = 0.1272..
            # P(Data=East|x=0.4) = 6/11 / 5 = 6/55 = 0.1090..
            # P(Data=East|x=0.5) = 5/11 / 5 = 1/11 = 0.0909..
            # P(Data=East|x=0.6) = 4/11 / 5 = 4/55 = 0.0727..
            # P(Data=East|x=0.7) = 3/11 / 5 = 3/55 = 0.0545..
            # P(Data=East|x=0.8) = 2/11 / 5 = 2/55 = 0.0363..
            # P(Data=East|x=0.9) = 1/11 / 5 = 1/55 = 0.0181..
            # P(Data=East|x=1.0) = 0 / 5 =           0
            #
            # It is then trivial to see that P(Data=West|Hypothesis X-dir) would be the same numbers, only in the reversed order:
            #
            # P(Data=West|x=0) = 0 / 5 =             0
            # P(Data=West|x=0.1) = 1/11 / 5 = 1/55 = 0.0181..
            # P(Data=West|x=0.2) = 2/11 / 5 = 2/55 = 0.0363..
            # P(Data=West|x=0.3) = 3/11 / 5 = 3/55 = 0.0545..
            # P(Data=West|x=0.4) = 4/11 / 5 = 4/55 = 0.0727..
            # P(Data=West|x=0.5) = 5/11 / 5 = 1/11 = 0.0909..
            # P(Data=West|x=0.6) = 6/11 / 5 = 6/55 = 0.1090..
            # P(Data=West|x=0.7) = 7/11 / 5 = 7/55 = 0.1272..
            # P(Data=West|x=0.8) = 8/11 / 5 = 8/55 = 0.1454..
            # P(Data=West|x=0.9) = 9/11 / 5 = 9/55 = 0.1636..
            # P(Data=West|x=1.0) = 10/11 / 5 = 2/11 = 0.1818..
            #
            # It is also trivial to see that P(Data=North|Hypothesis Y-dir) and P(Data=South|Hypothesis Y-dir) would be of a similar form, since we are dealing with a square:
            #
            # P(Data=North|y=0) = 10/11 / 5 = 2/11 = 0.1818..
            # P(Data=North|y=0.1) = 9/11 / 5 = 9/55 = 0.1636..
            # P(Data=North|y=0.2) = 8/11 / 5 = 8/55 = 0.1454..
            # P(Data=North|y=0.3) = 7/11 / 5 = 7/55 = 0.1272..
            # P(Data=North|y=0.4) = 6/11 / 5 = 6/55 = 0.1090..
            # P(Data=North|y=0.5) = 5/11 / 5 = 1/11 = 0.0909..
            # P(Data=North|y=0.6) = 4/11 / 5 = 4/55 = 0.0727..
            # P(Data=North|y=0.7) = 3/11 / 5 = 3/55 = 0.0545..
            # P(Data=North|y=0.8) = 2/11 / 5 = 2/55 = 0.0363..
            # P(Data=North|y=0.9) = 1/11 / 5 = 1/55 = 0.0181..
            # P(Data=North|y=1.0) = 0 / 5 =           0
            #
            # P(Data=South|y=0)_X = 0 / 5 =             0
            # P(Data=South|y=0.1)_X = 1/11 / 5 = 1/55 = 0.0181..
            # P(Data=South|y=0.2)_X = 2/11 / 5 = 2/55 = 0.0363..
            # P(Data=South|y=0.3)_X = 3/11 / 5 = 3/55 = 0.0545..
            # P(Data=South|y=0.4)_X = 4/11 / 5 = 4/55 = 0.0727..
            # P(Data=South|y=0.5)_X = 5/11 / 5 = 1/11 = 0.0909..
            # P(Data=South|y=0.6)_X = 6/11 / 5 = 6/55 = 0.1090..
            # P(Data=South|y=0.7)_X = 7/11 / 5 = 7/55 = 0.1272..
            # P(Data=South|y=0.8)_X = 8/11 / 5 = 8/55 = 0.1454..
            # P(Data=South|y=0.9)_X = 9/11 / 5 = 9/55 = 0.1636..
            # P(Data=South|y=1.0)_X = 10/11 / 5 = 2/11 = 0.1818..
            #
            # We also note that if the data given is neither north/south or east/west, but 0, which indicates a 
            # similar position a long one of the axes, equating to P(Data=Similar Position(0)|Hypothesis X-dir or Y-dir),
            # no new information is added, and the data probability given any hypothesis effectively becomes uniform (1/11)
            #
            # Thus, we have deduced the conditional data probabilities P(West|Hypothesis X-dir), P(East|Hypothesis X-dir), P(South|Hypothesis Y-dir), and P(North|Hypothesis Y-dir),
            # and can implement this programtically
            
            # Calculate the raw conditional probabilities
            ConditionalProbabilities = np.array([i / (len(PriorX[0])) for i in range(len(PriorX[0]))])
            # Normalize such that it sums to 1
            ConditionalProbabilities /= sum(ConditionalProbabilities)
            
            # Make the conditional probabilities run in both directions
            GradientPositive = ConditionalProbabilities
            GradientNegative = ConditionalProbabilities[::-1]
            
            # Make an array with probabilities for the case where either Y-dir or X-dir of additional ball is similar to first ball
            SameSpot = np.ones(len(PriorX[0])) / len(PriorX[0])
            
            # Make matrices with same size as the grid from the conditional probabilities given in (GradientPositive and GradientNegative)
            
            # For the X-direction, we only need to replicate the array in 10 additional rows
            RightofFirstMatrix = np.tile(GradientNegative, (len(PriorX[0]), 1)) # P(East|Hypothesis X-dir)
            LeftofFirstMatrix = np.tile(GradientPositive, (len(PriorX[0]), 1))  # P(West|Hypothesis X-dir)
            XSamePlacementMatrix = np.tile(SameSpot, (len(PriorX[0]), 1))       # P(Similar Location|Hypothesis X-dir)
            
            # For the Y-direction, we need to transpose the gradients, before replicating them 10 additional times
            AboveofFirstMatrix = np.tile(GradientNegative.reshape(len(PriorX[0]), 1), (1, len(PriorX[0]))) # P(North|Hypothesis Y-dir)
            BelowofFirstMatrix = np.tile(GradientPositive.reshape(len(PriorX[0]), 1), (1, len(PriorX[0]))) # P(South|Hypothesis Y-dir)
            YSamePlacementMatrix = np.tile(SameSpot.reshape(len(PriorX[0]), 1), (1, len(PriorX[0])))       # P(Similar Location|Hypothesis Y-dir)

            # We can now assign the conditional probabilities with simple if statements
            
            # P(East|Hypothesis X-dir)
            if AdditionalBalls[allAdditionalBalls, 0] == 1:
                DataProbability_X = RightofFirstMatrix
            # P(West|Hypothesis X-dir)    
            elif AdditionalBalls[allAdditionalBalls, 0] == -1:
                DataProbability_X = LeftofFirstMatrix
            # P(Similar Location|Hypothesis X-dir)    
            elif AdditionalBalls[allAdditionalBalls, 0] == 0:
                DataProbability_X = XSamePlacementMatrix
 
            # P(North|Hypothesis Y-dir)
            if AdditionalBalls[allAdditionalBalls, 1] == 1:
                DataProbability_Y = AboveofFirstMatrix
            # P(South|Hypothesis Y-dir)    
            elif AdditionalBalls[allAdditionalBalls, 1] == -1:
                DataProbability_Y = BelowofFirstMatrix
            # P(Similar Location|Hypothesis Y-dir)    
            elif AdditionalBalls[allAdditionalBalls, 1] == 0:
                DataProbability_Y = YSamePlacementMatrix

            # Since the matrices with the conditional data probabilities have been replicated x 11, we need to do an additional normalizing
            DataProbability_X /= np.sum(DataProbability_X)
            DataProbability_Y /= np.sum(DataProbability_Y)
            
            # We can then use Bayes theorem, specifically the proportionality P(Hypothesis|Data) ∝ P(Data|Hypothesis)*P(Hypothesis)
            # to update our belief
            P_hypothesis_given_data_X = DataProbability_X * PriorX
            P_hypothesis_given_data_Y = DataProbability_Y * PriorY
            
            # The updated belief then needs to be normalized
            # These updated belief are then feed back again in the top of the function, now acting as priors for the next iteration.
            P_hypothesis_given_data_X_normalized = P_hypothesis_given_data_X / np.sum(P_hypothesis_given_data_X)
            P_hypothesis_given_data_Y_normalized = P_hypothesis_given_data_Y / np.sum(P_hypothesis_given_data_Y)

            # When visualizing the probabilities dynamically, we finally combine the probabilities for the X and Y direction.
            # This is done by selecting the minimum probability at each grid from each of the normalized probability matrices describing a distinct direction.
            VisualizeProbabilities = np.minimum(P_hypothesis_given_data_X_normalized, P_hypothesis_given_data_Y_normalized)
            
            # The final probability matrix is then once more normalized to assure the probabilities sum to 1
            VisualizeProbabilities /= np.sum(VisualizeProbabilities)

            # Only plot when function in plotting mode
            if plot:
                # Update colors to be represented by the probabilities
                cax_1 = ax_1.pcolor(X, Y, VisualizeProbabilities, cmap='jet', shading='auto')
                cax_1.set_clim(0, np.max(VisualizeProbabilities))
                cbar.update_normal(cax_1)
                ax_1.set_xlabel('X')
                ax_1.set_ylabel('Y')
                
                
                
                # Remove patches after each additional ball (patches are added in the code below)
                # This gives the magenta square (frequentist ratio) dynamic behavior
                for patch in ax_1.patches:
                    patch.remove()

                # Add the white square (position of first ball) and subtract gridspacingForSquares for proper positioning of square
                ax_1.add_patch(plt.Rectangle((FirstBall_xy[0] - gridspacingForSquares, FirstBall_xy[1] - gridspacingForSquares), cell_width, cell_height, edgecolor='white', fill=False, linewidth=4))
            

            # The code below calculates the position of the ball according the frequentist method.
            # When calculating the total balls, one need to exclude all balls obtaining the same X or Y position as the first ball, indicated by 0
            total_current_number_of_balls_X = np.count_nonzero(AdditionalBalls[:allAdditionalBalls + 1, 0] != 0)
            total_current_number_of_balls_Y = np.count_nonzero(AdditionalBalls[:allAdditionalBalls + 1, 1] != 0)
            if total_current_number_of_balls_X != 0 and total_current_number_of_balls_Y != 0:
                # 1 - the ratio of balls to the east relative to total balls so far in the experiement determines X-cor.
                # 1 - the ratio of balls to the north relative to total balls so far in the experiement determines Y-cor.
                x_freq = 1 - len(np.where(AdditionalBalls[:allAdditionalBalls + 1, 0] == 1)[0]) / total_current_number_of_balls_X
                y_freq = 1 - len(np.where(AdditionalBalls[:allAdditionalBalls + 1, 1] == 1)[0]) / total_current_number_of_balls_Y
                
                # Additionally, we already know that balls obtain discrete coordinates given by X and Y obtaining any value 0.0, 0.1, 0.2, ... , 1.
                # Therefore, for the frequentist view, the x_start's and y_start's closest discrete point a ball can be positioned will be the grid
                # with highest probability of having the original ball.          
                x_freq_discrete = x[np.argmin(np.abs(x - x_freq))]
                y_freq_discrete = y[np.argmin(np.abs(y - y_freq))]
            else:
                # If either X-dir or Y-dir have only additional balls with similar coordinate as the first ball, then one cannot infer the direction at all, and
                # and the best estimate is just guessing the coordinate at random. 
                if total_current_number_of_balls_X == 0 and total_current_number_of_balls_Y == 0:
                    x_freq_discrete, y_freq_discrete = RandomBallCoordinates()
                    # To ensure compatability with the plot titles dependent on x_freq and y_freq
                    x_freq = x_freq_discrete
                    y_freq = y_freq_discrete
                
                # Only X-dir have only zeros    
                elif total_current_number_of_balls_X == 0 and total_current_number_of_balls_Y != 0:
                    x_freq_discrete, _  = RandomBallCoordinates()
                    # To ensure compatability with the plot titles dependent on x_freq and y_freq
                    x_freq = x_freq_discrete

                    # Normal calculation for Y-dir
                    y_freq = 1 - len(np.where(AdditionalBalls[:allAdditionalBalls + 1, 1] == 1)[0]) / total_current_number_of_balls_Y
                    y_freq_discrete = y[np.argmin(np.abs(y - y_freq))]
                
                # Only Y-dir have only zeros 
                elif total_current_number_of_balls_X != 0 and total_current_number_of_balls_Y == 0:    
                    y_freq_discrete, _  = RandomBallCoordinates()
                    # To ensure compatability with the plot titles dependent on x_freq and y_freq
                    y_freq = y_freq_discrete
                    
                    # Normal calculation for X-dir
                    x_freq = 1 - len(np.where(AdditionalBalls[:allAdditionalBalls + 1, 0] == 1)[0]) / total_current_number_of_balls_X
                    x_freq_discrete = x[np.argmin(np.abs(x - x_freq))]
            
        
        
            # Only plot when function in plotting mode
            if plot: 
                # Add square representing the inferred location of the first ball from the frequentist method
                # gridspacingForSquares is subtracted from the coordinates to place the square with the mid point at the inferred location
                ax_1.add_patch(plt.Rectangle((x_freq_discrete - gridspacingForSquares, y_freq_discrete - gridspacingForSquares), cell_width, cell_height, edgecolor='magenta', fill=False, linewidth=2))
                
                # Add the title of subplot
                string_title_1 = f'Bayesian inference after additional ball number: {allAdditionalBalls + 1}'
                string_title_2 = 'Position first ball (white); Frequentist/ratio approach (magenta)'
                string_title_3 = f"Position given by Frequentist/ratio: X = {round(x_freq, 3)}, Y = {round(y_freq, 3)}"
                ax_1.set_title(f'{string_title_1}\n{string_title_2}\n{string_title_3}')

            # Only use when function in plotting mode
            if plot: 
                plt.pause(0.1)  # Pause to update the plot

            # This can be used to stop the inference engine when bayesian probability have reached a given threshold
            #if np.max(VisualizeProbabilities) > 0.9999:
            #    break
            
            # After first button click, this is set to true, assuring that when clicking the button again, a new random location for the first ball is selected
            AlreadyClickedButton = True
            
            
            # Here we calculate the euclidean distance between the placement of first ball and the grid with maximum probability from bayesian and frequentist method
            
            # The coordinates for frequentist are given by x_freq_discrete and y_freq_discrete
            distance_Frequentist = np.sqrt((x_freq_discrete - FirstBall_xy[0])**2 + (y_freq_discrete - FirstBall_xy[1])**2)
            
            # Obtaining coordinates for grid with maximum probability given by bayes approach
            max_index = np.unravel_index(np.argmax(VisualizeProbabilities, axis=None), VisualizeProbabilities.shape)
            max_y, max_x = max_index  # Note: row index first, then column index
            # Retrieve the corresponding x and y coordinates from the meshgrid
            max_x_coord = X[max_y, max_x]
            max_y_coord = Y[max_y, max_x]
            # Calculate the distance
            distance_Bayes = np.sqrt((max_x_coord - FirstBall_xy[0])**2 + (max_y_coord - FirstBall_xy[1])**2)
        
            # Store the distances
            StoreEuclideanDistances.append([distance_Bayes, distance_Frequentist])
            
            

        # Only use when function in plotting mode
        if plot: 
            plt.draw()
        
        return StoreEuclideanDistances # Function returns these distances
    
    
    # This function is used to calculate the difference between the bayesian and frequentist approach, utilizing 1000 random iterations
    # for different amounts of additional balls in the set S = {1, 2, ..., 200}
    # Monte Carlo Simulation function
    def MonteCarloSimulation(MC_simulations, HowManyNewBalls, event=None):
        # Initiate variable and lists
        numberOfMCSimulations = MC_simulations
        TotalNumberOfBalls = HowManyNewBalls
        all_distances = []
        avg_distances_Bayes = []
        avg_distances_Frequentist = []

        # Loops through all iterations
        for numSim in range(1, numberOfMCSimulations+1):
            # Prints every 10th
            if numSim % 10 == 0:
                print(f"Simulation {numSim}/{numberOfMCSimulations}")
                
                ax_2.clear()
                ax_2.set_xlabel('Number of Additional Balls')
                ax_2.set_ylabel('Distance to First Ball')
                ax_2.set_title(f'MC simulation showing the average Euclidean distance between\nBayesian (blue) and Frequentist (orange) inference and location of the first ball\nafter {numberOfMCSimulations} MC-iterations for each specific number of additional balls')
                
                ax_2.annotate(
                    f"Simulation {numSim}/{numberOfMCSimulations}. Please be patient..",
                    xy=(0.5, 0.5), xycoords='axes fraction',
                    fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='green', alpha=0.5))
                plt.pause(0.001)
                plt.draw()
                
            
            # run_single_inference(numberOfBalls, plot=False) outputs a list of size 200, 2.
            # 200 elements corresponding to additional balls 1 to 200, with two elements for each
            # corresponding to distance from bayesian inference and distance from frequentist inference
            Bayes_Frequentist_distances = run_single_inference(TotalNumberOfBalls, plot=False)
            
            # For every MC-iteration, this is appended to all_distances
            all_distances.append(Bayes_Frequentist_distances)
        
        # Convert to array
        all_distances = np.array(all_distances)  # Shape: (1000, 200, 2)

        # clear axes before new figures.
        ax_2.clear()
        # Loop through the array with the specific number of additional balls, which should have 1000 different outcomes from the simulation
        # Average these outcomes, and update the plot.
        for numberOfBalls in range(1, len(all_distances[1 , :, 1])+1):
            
            # Take the average of all runs witch similar amount of balls, and append to the lists
            avg_distances_Bayes.append(np.mean(all_distances[:, numberOfBalls-1, 0], axis=0))
            avg_distances_Frequentist.append(np.mean(all_distances[:, numberOfBalls-1, 1], axis=0))
            
            # Running MSE for bayesian estimate using NumPy
            bayesian_MSE = np.mean(np.square(avg_distances_Bayes))
            
            # Running MSE for frequentist estimate using NumPy
            frequentist_MSE = np.mean(np.square(avg_distances_Frequentist))

            # Update the plot and axes
            ax_2.clear()
            ax_2.plot(range(1, numberOfBalls + 1), avg_distances_Bayes, color='blue', linewidth=2, label='Bayesian')
            ax_2.plot(range(1, numberOfBalls + 1), avg_distances_Frequentist, color='orange', linewidth=2, label='Frequentist')
            ax_2.set_xlabel('Number of Additional Balls')
            ax_2.set_ylabel('Distance to First Ball')
            ax_2.set_title(f'MC simulation showing the average Euclidean distance between\nBayesian (blue) and Frequentist (orange) inference and location of the first ball\nafter {numberOfMCSimulations} MC-iterations for each specific number of additional balls ({numberOfBalls}/{TotalNumberOfBalls})')
            
            # Annotate the MSE estimates
            ax_2.annotate(
                    f"Running MSE bayesian   = {np.round(bayesian_MSE, 5)} \n Running MSE frequentist = {np.round(frequentist_MSE, 5)}",
                    xy=(0.55, 0.85), xycoords='axes fraction',
                    fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
            
            # Show a special message at certain points
            if numberOfBalls >= 50:
                ax_2.annotate(
                    "If you made it this far, you probably already \n know that a Bayesian and frequentist approach\nproduces more or less the same accuracy of inference \n when all information is incorporated in both methods.\n (Bayesian somewhat favorable when data is sparse).",
                    xy=(0.55, 0.70), xycoords='axes fraction',
                    fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
            
            if numberOfBalls >= 100:
                ax_2.annotate(
                    "However, this is generally not the case in science. \n Usually, a p-value is calculated only based \n on the data investigated, without weighing\n the prior evidence of the hypothesis in question.\n\n Why else do we have a replication crisis in science?",
                    xy=(0.55, 0.46), xycoords='axes fraction',
                    fontsize=12, ha='center', va='center', bbox=dict(facecolor='red', alpha=0.2))    

            # Only initiate legends once
            if numberOfBalls >= 1:
                ax_2.legend()
                
            # Allow time to draw dynamically on figures
            plt.pause(0.05)
            plt.draw()
        
        
        
        
        
        
    # This initilizes the inference engine
    
    # Size of grid determines the amount of locations
    AmountOfLocations = HowManyGridPositionsInEachDirection
    
    # Array showing all possible X/Y coordinates for balls: 0, 0.1, 0.2, ... , 1
    randomArray = np.linspace(0, 1, AmountOfLocations)
    
    # Choosing coordinates for first ball
    FirstBall_xy = [0,0]
    FirstBall_xy[0], FirstBall_xy[1] = RandomBallCoordinates()
    x_first = FirstBall_xy[0]
    y_first = FirstBall_xy[1]
    
    # This decides wether to use the first ball position assigned before the press of the start button or generate new random numbers for first ball
    AlreadyClickedButton = False
    
    # Coordinates for the axes
    x = np.linspace(0, 1, AmountOfLocations)
    y = np.linspace(0, 1, AmountOfLocations)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the gridspace this amount too and divide by 2. This is to paint the squares properly
    gridspacingForSquares = (x[1] - x[0]) / 2

    # Figure with two subplots
    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(bottom=0.2, wspace=0.3)
    
    # Setup for the first subplot which will show the inference engine in action
    ax_1.set_xlabel('X')
    ax_1.set_ylabel('Y')
    string_title_1 = 'Bayesian inference after additional ball number: 0'
    string_title_2 = 'Position first ball (white square); Bernoulli/ratio approach (magenta square)'
    string_title_3 = "Position given by Frequentist/ratio: X = ?, Y = ?"
    ax_1.set_title(f'{string_title_1}\n{string_title_2}\n{string_title_3}')
    
    
    # When solving the thought experiment, we will divide the probabilities in two. One probability represents the X-dir, and the other the Y-dir.
    # The initial probabilities of the first ball for the Y (south-north) direction starts at 1/11 for the (since it can obtain 11 positions in this direction) and
    # same for the X (west-east) direction.
    # Both initial_value and Z will be used for further calculation in run_single_inference()
    
    # Initial probability representing the X-dir and Y-dir
    initial_value = 1 / AmountOfLocations
    
    # Initial probabilities in coordinates format
    Z = initial_value * np.ones(X.shape)
    
    # However, when visualizing the correct probability of the whole X*Y grid, these need to be normalized (as they are now only representing the probabilities in discrete directions (X-dir, Y-dir)).
    # Normalization should ultimately yield 1/121 initial (prior) 
    # probability for each grid containing the first ball, simply because there are 121 discrete positions the first ball can take with equal probabilities (random assignment).
    Z_normalized = Z
    Z_normalized /= np.sum(Z_normalized)

    # Plot the initial colors for the grid which represents the initial probabilities (priors)
    cax_1 = ax_1.pcolor(X, Y, Z_normalized, cmap='jet', shading='auto')
    cbar = fig.colorbar(cax_1, ax=ax_1, label='Bayesian Inference Probabilities:\nP(Hypothesis | Data)', orientation='vertical')

    # Draw a white square at the position of the first ball to be inferred
    cell_width = x[1] - x[0]
    cell_height = y[1] - y[0]
    
    # Add square representing the location of the first ball
    # gridspacingForSquares is subtracted from the coordinates to place the square with the mid point at the location
    ax_1.add_patch(plt.Rectangle((x_first - gridspacingForSquares, y_first - gridspacingForSquares), cell_width, cell_height, edgecolor='white', fill=False, linewidth=4))
    
    

    # Setup the second subplot for the MC-simulation calculating the accuracy and differences between the Bayesian and Frequentist approach for additional balls in range 1 to 500
    ax_2.set_xlabel('Number of Additional Balls')
    ax_2.set_ylabel('Distance to First Ball')
    ax_2.set_title(f'MC simulation showing the average Euclidean distance between\nBayesian (blue) and Frequentist (orange) inference and location of the first ball\nafter {MC_simulations} MC-iterations for each specific number of additional balls')





    # Buttons for both subplots

    # Using functools.partial to pass additional arguments to first button
    callback_with_args_btn1 = partial(run_single_inference, HowManyNewBalls)
    
    # Button to run inference enginge
    ax_1_button = plt.axes([0.125, 0.05, 0.27, 0.075])
    btn_1 = Button(ax_1_button, '1. Start bayesian inference engine!')
    btn_1.on_clicked(callback_with_args_btn1)
    
    # Using functools.partial to pass additional arguments to second button
    callback_with_args_btn2 = partial(MonteCarloSimulation, MC_simulations, HowManyNewBalls)
    
    # Button to run MC simulation comparing accuracy of bayesian and frequentist inference for different number of additional balls (additional data)
    ax_2_button = plt.axes([0.575, 0.05, 0.315, 0.075])
    btn_2 = Button(ax_2_button, '2. Start MC-simulation comparing bayesian and frequentist!')
    btn_2.on_clicked(callback_with_args_btn2)

    plt.draw()
    plt.show()

# Here you can choose the amount of balls for the inference engine (subplot 1)
# and the amount of iterations for the Monte Carlo Simulation (subplot 2)
# Size of the grid can also be choosen by HowManyGridPositionsInEachDirection.
# At 11, this means the resulting grid will be 11x11. Remember that increasing
# the grid size will increase the amount of computations exponentially, making the
# code run alot slower.
bayesian_vs_bernoulli(HowManyNewBalls=200, MC_simulations=1000, HowManyGridPositionsInEachDirection=11)
