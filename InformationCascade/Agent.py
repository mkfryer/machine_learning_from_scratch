import numpy as np, numpy.random

# class DistributionAbstract:
#     """ """

#     def __init__(self):
#         """ """

#     def draw():
#         raise NotImplementedError("Must be implimented!")


# class BetaDistribution(DistributionAbstract):
#     """ """

class Agent:
    """ 
    Represents a household in the cape town crisis that must decide which one of 
    three water wells to choose to draw from

    variables:
    dist_params - ndarray (3 x 1) - the parameters of the tri-noulli distribution
    """

    def __init__(self):
        #prior distribution parameters
        self.dist_params = np.random.random(3)
        self.dist_params /= np.sum(self.dist_params)


    def get_MLE(observations):
        """
        Paramters:
        observations - ndarray (n x 3):  

        Returns ndarray (n x 3):  most likely parameters given the observations
        """
        # liklihood = lambda x, theta: theta[0]**x[0] * theta[1]**x[1] * theta[2]**x[2]
        

    def update_dist_params(observzaations):
        """
        Do something with mle ... 
        """


    # def act(self):
    #     if len(self.signals) == 0:

    #         return np.argmax(prior_knowledge)

    #     else:

a = Agent()
print(a.dist_params)
print(sum(a.dist_params))




# a_executions = 0
# b_executions = 0
# c_executions = 0
# b_pardon = 0
# c_pardon = 0
# P = {"b": {"a":0, "c" : 0}, "c": {"a":0, "b" : 0}}

# for _ in range(1000):
#     report = np.random.random(1)[0]
#     #a will be executed
#     if report <= .33:
#         #b or c equally likely to get pardon
#         if np.random.random(1)[0] > .5:
#             P["b"]["a"] += 1
#             b_pardon += 1
#         else:
#             P["c"]["a"] += 1
#             c_pardon += 1
#         a_executions += 1
#     #b will be executed
#     elif report <= .66:
#         P["c"]["b"] += 1
#         c_pardon += 1
#         b_executions += 1
#     #c will be executed
#     else:
#         P["b"]["c"] += 1
#         b_pardon += 1
#         c_executions += 1

# print(P)

# print("a ", a_executions/1000)
# print("b ", b_executions/1000)
# print("c ", c_executions/1000)

# print("b pardons", b_pardon/1000)
# print("c pardons", c_pardon/1000)