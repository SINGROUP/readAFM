# commands to execute with matplotlib turned on

# make this a class or function after resolving the matplotlib issues


import numpy as np


batchsize = 50                                 # set manually!

view_sol=open('view_solution_*.dat','w')       # set stars manually?
view_pred=open('view_prediction_*.dat','w')    # set stars manually?

solution = np.load(view_sol)
prediction = np.load(view_pred)

# This won't work this way, but it should be an inspiration on how to view the atoms

for i in range(batchsize):
    for atom in range(5):
        imshow(prediction(i,:,:,atom)-solution(i,:,:,atom))
