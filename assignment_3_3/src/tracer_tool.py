import igl
import scipy.spatial as ds
import numpy as np
import math
import numpy.linalg as la
from tracer_helper import Mesh, rotate_vector
from tracer_utils import intersection_event, asymptotic_path

class AsymptoticTracer:

    def __init__(self, filename, step_size=0.01, num_steps=200):
        self.mesh = Mesh(filename)
        self.step_size = step_size
        self.num_steps = num_steps
        # Variables for visualisation
        self.flagA = False
        self.flagB = False
        self.flag_intersections = False
        # Main data
        self.paths = [np.empty((0,3), float) for i in range(2)] 
        self.paths_indexes = [[] for i in range(2)] 
        self.paths_flatten = [[] for i in range(2)] 
        self.paths_lengths = [[] for i in range(2)] 
        self.paths_labels = [[] for i in range(2)]
        self.intersections = [[] for i in range(2)] 
        self.samples_indexes = [[] for i in range(2)]
        self.intersection_points = np.empty((0,3), float)
    
    def delete_path(self, path_index, first_principal_direction=True):
        self.paths_indexes[1-first_principal_direction].pop(path_index)
        self.paths_flatten[1-first_principal_direction].pop(path_index)
        self.paths_lengths[1-first_principal_direction].pop(path_index)
        self.paths_labels[1-first_principal_direction].pop(path_index)
        self.intersections[1-first_principal_direction].pop(path_index)
        self.samples_indexes[1-first_principal_direction].pop(path_index)

    def num_pathsA(self):
        return len(self.paths_indexes[0])

    def num_pathsB(self):
        return len(self.paths_indexes[1])

    def generate_asymptotic_path(self, vertex_idx, first_principal_direction, num_neighbors, sampling_dist):
    
        P,A,PP = asymptotic_path(vertex_idx, self.mesh, self.num_steps, self.step_size, first_principal_direction, num_neighbors, sampling_dist)

        if len(P)>0:
            start_vertex = len(self.paths[1-first_principal_direction])
            self.paths[1-first_principal_direction] = np.append(self.paths[1-first_principal_direction], P, axis=0)
            self.paths_indexes[1-first_principal_direction].append(np.arange(start_vertex, start_vertex+len(P)))
            # Initialize data for flattening
            self.intersections[1-first_principal_direction].append(np.empty((0,3), float))
            self.paths_lengths[1-first_principal_direction].append(np.zeros(len(P)-1))
            self.paths_flatten[1-first_principal_direction].append(np.empty((0,2), float))
            self.paths_labels[1-first_principal_direction].append(np.empty((0,2), float))
            self.flag_intersections = False
        
        return P, PP

    def generate_intersection_network(self):
        treeA = ds.KDTree(self.paths[0])
        treeB = ds.KDTree(self.paths[1])

        strips_numA = len(self.paths_indexes[0])
        strips_numB = len(self.paths_indexes[1])
        strips_num = strips_numA if strips_numA > strips_numB else strips_numB

        self.intersection_points = np.empty((0,3), float)

        for i in range(strips_num):
            if i< strips_numA:
                self.generate_intersection_path(i, treeB, False)
            if i< strips_numB:
                self.generate_intersection_path(i, treeA, True)

        self.flag_intersections = True
        return self.intersection_points

    def generate_intersection_path(self, pathIndex, tree, second_direction=False):

        self.intersections[second_direction][pathIndex] = np.empty((0,3), float)
        path_indexesA = self.paths_indexes[second_direction][pathIndex]
        self.paths_lengths[second_direction][pathIndex] = np.zeros(len(path_indexesA)-1)

        for i in range (len(path_indexesA)-1):
            
            # Get the segment on the first path
            origA = self.paths[second_direction][path_indexesA[i]]
            endA = self.paths[second_direction][path_indexesA[i+1]]
            vecA = endA - origA
                
            # Store the length of the edge
            self.paths_lengths[second_direction][pathIndex][i] = la.norm(vecA)

            for j in range(len(self.paths_indexes[1-second_direction])):

                # Get the KDTree
                path_indexesB = self.paths_indexes[1-second_direction][j]

                # Query for closest nodes on the second path (Find closest nodes for both ends)
                dist, closest_idxB = tree.query([origA,endA], 1)

                # Find intersections in both end-nodes
                count = len(path_indexesB)

                for idx in closest_idxB: 
                    origB = tree.data[idx]

                    # Check for intersection events with the node following the closest node on the second path
                    # Break if an intersection is found
                    if idx in path_indexesB:

                        local_idx = np.where(path_indexesB==idx)[0][0]

                        vecB = None
                        
                        if local_idx<=count-2:
                            vecB = tree.data[path_indexesB[local_idx+1]] - origB
                            t, u, inter = intersection_event(origA, vecA, origB, vecB)
                            if inter==1 and t>=0 and t<=1 and u>=0 and u<=1:
                                # Store edge index and the parameter where the intersection occured. 
                                p = origA + vecA * t

                                distance = (self.intersection_points[:,0]-p[0])**2 + (self.intersection_points[:,1]-p[1])**2 + (self.intersection_points[:,2]-p[2])**2
                                closest_intersection = np.where(distance<1e-3)[0]

                                if len(closest_intersection)==0: # No duplicate intersection
                                    label = len(self.intersection_points)
                                    self.intersections[second_direction][pathIndex] = np.append(self.intersections[second_direction][pathIndex],np.array([[i,t,label]]), axis=0)                     
                                    self.intersection_points = np.append(self.intersection_points, np.array([p]), axis=0)
                                    break
                                else: # Duplicated intersection
                                    label = closest_intersection[0]
                                    self.intersections[second_direction][pathIndex] = np.append(self.intersections[second_direction][pathIndex],np.array([[i,t,label]]), axis=0)    
                                    break                 

                        
                        # If no intersection is found, check with the node preceding the closest node on the second path
                        # Break if an intersection is found
                        if local_idx >= 1:
                            vecB = tree.data[path_indexesB[local_idx-1]] - origB
                            t, u, inter = intersection_event(origA, vecA, origB, vecB)
                            if inter==1 and t>=0 and t<=1 and u>=0 and u<=1:
                                # Store edge index and the parameter where the intersection occured.
                                p = origA + vecA * t

                                distance = (self.intersection_points[:,0]-p[0])**2 + (self.intersection_points[:,1]-p[1])**2 + (self.intersection_points[:,2]-p[2])**2
                                closest_intersection = np.where(distance<1e-3)[0]

                                if len(closest_intersection)==0: # No duplicate intersection
                                    label = len(self.intersection_points)
                                    self.intersections[second_direction][pathIndex] = np.append(self.intersections[second_direction][pathIndex],np.array([[i,t,label]]), axis=0)                     
                                    self.intersection_points = np.append(self.intersection_points, np.array([p]), axis=0)
                                    break
                                else: # Duplicated intersection
                                    label = closest_intersection[0]
                                    self.intersections[second_direction][pathIndex] = np.append(self.intersections[second_direction][pathIndex],np.array([[i,t,label]]), axis=0)    
                                    break  






    