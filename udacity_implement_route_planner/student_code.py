def shortest_path(M, start, goal):
    
    openset = {start}
    closeset = set()
    f_value = dict()
    f_value[start] = dis_two_nodes(M.intersections[start], M.intersections[goal])
    g_value = dict()
    g_value[start] = 0.0
    parent_node = dict()
    path = []

    while len(openset) > 0:
        current_node = find_min_f_value(openset, f_value)

        if current_node == goal:
            path.insert(0, current_node)
            while current_node in parent_node:
                current_node = parent_node[current_node]
                path.insert(0, current_node)
            return path

        openset.remove(current_node)
        closeset.add(current_node)

        for nearby_node in M.roads[current_node]:
            if nearby_node in closeset:
                continue

            openset.add(nearby_node)
            temporary_g_value = g_value[current_node] + dis_two_nodes(M.intersections[current_node], M.intersections[nearby_node])            
            if nearby_node in g_value:
                if temporary_g_value >= g_value[nearby_node]:
                    continue
            
            parent_node[nearby_node] = current_node
            g_value[nearby_node] = temporary_g_value
            f_value[nearby_node] = g_value[nearby_node] + dis_two_nodes(M.intersections[nearby_node], M.intersections[goal])

    print("shortest path called")
    return

def dis_two_nodes(pos1, pos2):
    distance = (((pos1[0] - pos2[0])**2) + ((pos1[1] - pos2[1])**2))**(0.5)
    return distance    

def find_min_f_value(openset, f_value):
    min_f_value_node = 32768
    min_f_value = float('inf')
    for node in f_value:
        if node in openset:
            if f_value[node] < min_f_value:
                min_f_value_node = node
                min_f_value = f_value[node]
    return min_f_value_node