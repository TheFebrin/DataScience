from random import randint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def generate_points(no, max_x, max_y):
    '''
    There is 80% chance that point will be next to existing one +/- 1
    20% chance that it will be placed randomly
    '''
    POINTS = [(1,1), (max_x, max_y), (0, max_y), (max_x, 0), (max_x//2, max_y//2)]
    for _ in range(no):
        rand = randint(1,10)
        if rand >= 3:
            rand_point = randint(0, len(POINTS) - 1)
            chosen_x, chosen_y = POINTS[rand_point]
            new_x = chosen_x + randint(-1,1)
            new_y = chosen_y + randint(-1,1)
            POINTS.append((new_x, new_y))
        else:
            new_x = randint(0, max_x)
            new_y = randint(0, max_y)
            POINTS.append((new_x, new_y))

    return POINTS

def KNN(POINTS):
    pass

if __name__ == '__main__':
    number_of_points = 20
    x_max, y_max = 20, 20
    POINTS = np.array(generate_points(number_of_points, x_max, y_max))
    print(POINTS)

    sns.scatterplot(x='x', y='y', hue='y', data=pd.DataFrame(POINTS, columns=['x','y']))
    plt.show()
