import matplotlib.pyplot as plt 
import numpy as np


def scatter_plotter(data_set, x_axis='Run Number', y_axis='Score', name='The Score For Each Run'):

    # setting x and y values.
    x = np.arange(len(data_set))
    y = data_set

    # plotting the points  
    plt.scatter(x, y) 

    # naming the x axis 
    plt.xlabel(x_axis) 
    # naming the y axis 
    plt.ylabel(y_axis) 

    # giving a title to my graph 
    plt.title(name) 

    # function to show the plot 
    plt.show() 

  

  

  

  

