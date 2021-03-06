import pdb
from helpers import normalize, blur

def initialize_beliefs(grid):
    height = len(grid)
    width = len(grid[0])
    area = height * width
    belief_per_cell = 1.0 / area
    beliefs = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(belief_per_cell)
        beliefs.append(row)
    return beliefs

def sense(color, grid, beliefs, p_hit, p_miss):
    new_beliefs = []

    #
    # TODO - implement this in part 2
    #
    
    for i in range(0, len(beliefs)):
        for j in range(0, len(beliefs[i])):
            hit = (color == grid[i][j])
            beliefs[i][j] = beliefs[i][j] * (hit * p_hit + (1 - hit) * p_miss)
            #new_beliefs = [x for x in beliefs]
            
    s = sum([sum(beliefs[i]) for i in range(0, len(beliefs))])
    new_beliefs = beliefs
    
    for i in range(0, len(new_beliefs)):
        for j in range(0, len(new_beliefs[i])):
            new_beliefs[i][j] = new_beliefs[i][j] / s

    return new_beliefs

def move(dy, dx, beliefs, blurring):
    height = len(beliefs)
    width = len(beliefs[0])
    new_G = [[0.0 for i in range(width)] for j in range(height)]
    """
    for i, row in enumerate(beliefs):
        for j, cell in enumerate(row):
            new_i = (i + dy) % height
            new_j = (j + dx) % width
            #pdb.set_trace()
            new_G[int(new_i)][int(new_j)] = cell
    """
    for i in range(height):
        for j in range(width):
            new_i = (i + dy) % height
            new_j = (j + dx) % width
            #pdb.set_trace()
            new_G[int(new_i)][int(new_j)] = beliefs[i][j]

    return blur(new_G, blurring)