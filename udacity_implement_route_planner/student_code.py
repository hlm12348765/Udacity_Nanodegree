from helpers import Map

def shortest_path(M, start, goal):
    pos = M.intersections
    nearby = M.roads

    def dis_two_points(pos1, pos2):
        distance = (((pos1[0] - pos2[0])**2) + ((pos1[1] - pos2[1])**2))**(0.5)
        return distance  

    dis_nearby = {}
    for 

    print("shortest path called")
    return

    