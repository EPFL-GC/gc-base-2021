import numpy as np
import numpy.linalg as la
import svgwrite
from tracer_tool import AsymptoticTracer

class FabricationHelper:

    def __init__(self, width, thickness, spacing, scale_length=1.0):
        self.strips_width = width
        self.strips_spacing = spacing
        self.strips_thickness = thickness
        self.scale_length = scale_length
        self.flag_flatten = False

    def generate_flatten_path(self, pathIndex, second_direction=False, position=np.array([0,0,0])):

        length_direction = np.array([1,0,0])
        width_direction = np.array([0,1,0]) * self.strips_width * 0.5
        if second_direction:
           width_direction *= -1
        
        self.intersection_points[second_direction][pathIndex] = np.empty((0,3), float)
        self.intersection_labels[second_direction][pathIndex] = []

        intersections = self.intersections[second_direction][pathIndex]
        paths_lengths = self.paths_lengths[second_direction][pathIndex]
        length = np.trace(np.diag(paths_lengths)) * self.scale_length

        path_forwards = np.empty((0,3), float)
        path_backwards = np.empty((0,3), float)
        path_intersection = np.empty((0,3), float)

        # Add start point
        start_position_forward = position + width_direction
        start_position_backward = position - width_direction
        end_position_forward = position + width_direction + length_direction * length
        end_position_backward = position - width_direction + length_direction * length
        path_forwards = np.append(path_forwards, np.array([start_position_forward]),axis=0)
        path_backwards = np.append(path_backwards, np.array([start_position_backward]),axis=0)

        # Connections
        if len(intersections)>0:
            for i in range(len(intersections)):

                intersection_event = intersections[i]
                # Store label
                self.intersection_labels[second_direction][pathIndex].append(int(intersection_event[2]))

                # Add intersection points
                edge_index = int(intersection_event[0])
                
                partial_length = (np.trace(np.diag(paths_lengths[:edge_index])) + paths_lengths[edge_index] * intersection_event[1]) * self.scale_length

                p = position + length_direction * partial_length

                path_intersection = np.append(path_intersection, np.array([p]), axis=0)

                p += width_direction

                vec_offset = length_direction * self.strips_thickness * 0.5

                if partial_length - self.strips_thickness * 0.5 <= 0:
                    p0 = position
                    p1 = p + vec_offset - width_direction
                    p2 = p + vec_offset
                    path_forwards = np.array([p0, p1, p2])
                elif partial_length + self.strips_thickness * 0.5 >= length:
                    p0 = p - vec_offset
                    p1 = p - vec_offset - width_direction
                    p2 = position + length_direction * length
                    path_forwards = np.append(path_forwards, np.array([p0, p1, p2]), axis=0)
                else:
                    p0 = p - vec_offset
                    p1 = p - vec_offset - width_direction
                    p2 = p + vec_offset - width_direction
                    p3 = p + vec_offset
                    path_forwards = np.append(path_forwards, np.array([p0, p1, p2, p3]), axis=0)

            if end_position_forward[0]>path_forwards[len(path_forwards)-1][0]:
                path_forwards = np.append(path_forwards, np.array([end_position_forward]), axis=0)
            path_backwards = np.append(path_backwards, np.array([end_position_backward]), axis=0)
        else:
            path_forwards = np.append(path_forwards, np.array([end_position_forward]), axis=0)
            path_backwards = np.append(path_backwards, np.array([end_position_backward]), axis=0)

        path = path_backwards[::-1]

        # Combine paths
        path = np.append(path, path_forwards, axis=0)
        path = np.append(path, np.array([path[0]]), axis=0)
        self.paths_flatten[second_direction][pathIndex] = path
        self.intersection_points[second_direction][pathIndex] = path_intersection

        return path, length

    def generate_flatten_network(self, tracer):
        self.paths_flatten = tracer.paths_flatten
        self.paths_lengths = tracer.paths_lengths
        self.intersection_points = tracer.paths_labels
        self.intersection_labels = [[] for i in range(2) ]
        self.intersections = tracer.intersections
        self.strips_numA = len(tracer.paths_indexes[0])
        self.strips_numB = len(tracer.paths_indexes[1])
        # init labels 
        self.intersection_labels[0] = [[] for i in range(self.strips_numA)]
        self.intersection_labels[1] = [[] for i in range(self.strips_numB)]

        strips_num = self.strips_numA if self.strips_numA > self.strips_numB else self.strips_numB

        # Generate flat strips
        strips_added = 0

        # Fabrication parameters
        self.board_width = self.strips_width * (self.strips_numA+self.strips_numB) + self.strips_spacing * (self.strips_numA+self.strips_numB+1)
        self.board_length = 0

        for i in range(strips_num):
            if i< self.strips_numA:
                position = np.array([ self.strips_spacing, self.strips_width * 0.5 + self.strips_spacing + (self.strips_width + self.strips_spacing) * strips_added, 0 ] )
                points, path_len = self.generate_flatten_path(i, False, position)
                strips_added +=1

                if path_len>self.board_length:
                    self.board_length = path_len
            
            if i< self.strips_numB:
                position = np.array([ self.strips_spacing, self.strips_width * 0.5 + self.strips_spacing + (self.strips_width + self.strips_spacing) * strips_added, 0 ] )
                points, path_len = self.generate_flatten_path(i, True, position)
                strips_added +=1

                if path_len>self.board_length:
                    self.board_length = path_len

        self.flag_flatten = True

    def generate_svg_file(self, filename, font_size=1, cutting_color="red", engraving_color="black", length=None, width=None):
        
        if length!=None:
            self.board_length = length
        if width!=None:
            self.board_width = width

        dwg = svgwrite.Drawing(filename,size=(str(self.board_length), str(self.board_width)), fill="none", stroke_width=".005cm")

        strips_numA = len(self.paths_flatten[0])
        strips_numB = len(self.paths_flatten[1])
        strips_num = strips_numA if strips_numA > strips_numB else strips_numB

        for i in range(strips_num):
            if i< strips_numA:
                pts = self.paths_flatten[0][i][:,[0,1]]
                dwg.add(dwg.polyline(pts, stroke=cutting_color))
                p0 = self.paths_flatten[0][i][0]
                p1 = self.paths_flatten[0][i][len(self.paths_flatten[0][i])-2]
                p2 = self.paths_flatten[0][i][len(self.paths_flatten[0][i])-3]
                vecX = p1-p2
                vecY = (p1-p0) * 0.5
                vecX /= la.norm(vecX)
                pos = p0 + vecY - vecX * self.strips_width * 0.9
                dwg.add(dwg.text('A'+str(i), insert=pos, fill=engraving_color, font_family="Code Light" ,font_size=str(font_size)))
                pts = self.intersection_points[0][i]
                labels = self.intersection_labels[0][i]
                for j in range(len(pts)):
                    pos = pts[j] - vecY * 0.5
                    dwg.add(dwg.text('c' + str(labels[j]), insert=pos, fill=engraving_color, font_family="Code Light" ,font_size=str(font_size * 0.7)))
                    
            if i< strips_numB:
                pts = self.paths_flatten[1][i][:,[0,1]]
                dwg.add(dwg.polyline(pts, stroke=cutting_color))
                p0 = self.paths_flatten[1][i][0]
                p1 = self.paths_flatten[1][i][len(self.paths_flatten[1][i])-2]
                p2 = self.paths_flatten[1][i][len(self.paths_flatten[1][i])-3]
                vecX = p1-p2
                vecY = (p1-p0) * 0.5
                vecX /= la.norm(vecX)
                pos = p0 + vecY - vecX * self.strips_width * 0.9
                dwg.add(dwg.text('B'+str(i), insert=pos, fill=engraving_color, font_family="Code Light" ,font_size=str(font_size)))
                pts = self.intersection_points[1][i]
                labels = self.intersection_labels[1][i]
                for j in range(len(pts)):
                    pos = pts[j] - vecY * 0.5
                    dwg.add(dwg.text('c' + str(labels[j]), insert=pos, fill=engraving_color, font_family="Code Light" ,font_size=str(font_size * 0.7)))
        #Board boundary
        points = np.array([[0.,0.],[self.board_length,0.],[self.board_length, self.board_width],[0., self.board_width],[0.,0.]])
        dwg.add(dwg.polyline(points, stroke=engraving_color))       
        dwg.save()
