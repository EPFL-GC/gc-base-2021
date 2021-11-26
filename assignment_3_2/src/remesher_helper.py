import gmsh
import math

def remesh(filename, mesh_size):
    # Import mesh
    file = open(filename)

    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("remesher")

    # The tags of the corresponding nodes:
    nodes = []
    # The x, y, z coordinates of all the nodes:
    coords = []
    # The connectivities of the triangle elements (3 node tags per triangle)
    tris = []
    # Tags
    node_tag = 1

    # Read file
    for line in file.readlines():
        data = [d.strip() for d in line.split(' ')]
        
        # Generatae nodes
        if(data[0]=="v"):
            nodes.append(node_tag)
            coords.extend([float(data[1]),float(data[2]),float(data[3])])
            node_tag +=1

        # Generate trinagular faces
        if(data[0]=="f"):
            # .obj from blender
            if '//' in data[1]:
                tris.extend([int([d.strip() for d in data[1].split('//')][0]),
                             int([d.strip() for d in data[2].split('//')][0]),
                             int([d.strip() for d in data[3].split('//')][0])])
                # Add second triangle if quad face
                if len(data) == 5:
                    tris.extend([int([d.strip() for d in data[3].split('//')][0]),
                    int([d.strip() for d in data[4].split('//')][0]),
                    int([d.strip() for d in data[1].split('//')][0])])
            elif '/' in data[1]:
                tris.extend([int([d.strip() for d in data[1].split('/')][0]),
                             int([d.strip() for d in data[2].split('/')][0]),
                             int([d.strip() for d in data[3].split('/')][0])])
                # Add second triangle if quad face
                if len(data) == 5:
                    tris.extend([int([d.strip() for d in data[3].split('/')][0]),
                    int([d.strip() for d in data[4].split('/')][0]),
                    int([d.strip() for d in data[1].split('/')][0])])
            # .obj general
            else:
                tris.extend([int(data[1]),int(data[2]),int(data[3])])
                # Add second triangle if quad face
                if len(data) == 5:
                    tris.extend([int(data[3]),int(data[4]),int(data[1])])


    model_tag = gmsh.model.addDiscreteEntity(2)
    # Add all the nodes on the surface (for simplicity... see below):
    gmsh.model.mesh.addNodes(2, model_tag, nodes, coords)
    # Triangle elements: Type 2 for 3-node triangle elements:
    gmsh.model.mesh.addElementsByType(model_tag, 2, [], tris)

    # Create underlying surface for meshing
    gmsh.model.mesh.classifySurfaces(math.pi, True, True, math.pi)
    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    # Meshing parameters
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    # Generate mesh
    gmsh.model.mesh.generate(2)

    node_tags, node_coords, node_param = gmsh.model.mesh.getNodes()
    element_tags, elements_node_tags = gmsh.model.mesh.getElementsByType(2)

    # End gmsh
    gmsh.finalize()

    # Get new nodes and faces
    v_remesh = node_coords.reshape((len(node_tags),3))
    f_remesh= elements_node_tags.reshape((len(element_tags),3))-1
    
    return v_remesh, f_remesh