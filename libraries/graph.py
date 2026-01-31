import numpy as np
import torch
import itertools
import sys
import os
import json

from pymatgen.core.structure import Structure
from scipy.spatial           import Voronoi
from rdkit                   import Chem

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_all_linked_tessellation(
        atomic_data,
        structure
):
    """Gets the distances by pairs of particles, considering images with periodic boundary conditions (PBC).

    Args:
        atomic_data        (dict):                      A dictionary with all node features.
        structure          (pymatgen Structure object): Structure from which the graph is to be generated
        distance_threshold (float, optional):           The distance threshold for edge creation (default is 6).

    Returns:
        nodes      (list): A tensor containing node attributes.
        edges      (list): A tensor containing edge indices.
        attributes (list): A tensor containing edge attributes (distances).
    """

    # Extract direct positions, composition and concentration as lists
    positions     = np.array([site.frac_coords for site in structure.sites])
    composition   = [element.symbol for element in structure.composition.elements]
    concentration = np.array([sum(site.species_string == element for site in structure.sites) for element in composition])

    # Counting number of particles
    total_particles = np.sum(concentration)

    # Generating graph structure, getting particle types
    particle_types = []
    for i in range(len(composition)):
        particle_types += [i] * concentration[i]

    # Adding nodes and edges.
    nodes = []
    edges = []
    attributes = []
    for index_0 in range(total_particles):
        # Get particle type (index of type wrt composition in POSCAR)
        particle_type = particle_types[index_0]

        # Name of the current species
        species_name = composition[particle_type]

        # Adding the nodes (mass, charge, electronegativity and ionization energies)
        nodes.append([atomic_data[species_name]['atomic_mass'],
                      atomic_data[species_name]['charge'],
                      atomic_data[species_name]['electronegativity'],
                      atomic_data[species_name]['ionization_energy']])

        # Get the initial position
        position_0 = positions[index_0]
        position_cartesian_0 = np.dot(position_0, structure.lattice.matrix)

        # Explore images of all particles in the system
        # Starting on index_0, thus exploring possible images with itself (except for i,j,k=0, exact same particle)
        for index_i in np.arange(index_0, total_particles):
            # Get the initial position
            position_i = positions[index_i]

            # Move to the corresponding image and convert to cartesian distances
            position_cartesian_i = np.dot(position_i, structure.lattice.matrix)

            # New distance as Euclidean distance between both reference and new image particle
            distance = np.linalg.norm([position_cartesian_0 - position_cartesian_i])

            # Append this point as an edge connection to particle 0
            edges.append([index_0, index_i])
            attributes.append([distance])
    return nodes, edges, attributes


def get_voronoi_tessellation(
        atomic_data,
        temp_structure,
        periodicity
):
    """
    Get the Voronoi nodes of a structure.
    Templated from the TopographyAnalyzer class, added to pymatgen.analysis.defects.utils by Yiming Chen, but now deleted.
    Modified to map down to primitive, do Voronoi analysis, then map back to original supercell; much more efficient.
    See commit 8b78474 'Generative models (basic example).ipynb'.

    Args:
        atomic_data    (dict):                      A dictionary with all node features.
        temp_structure (pymatgen Structure object): Structure from which the graph is to be generated.
        periodicity    (bool):                      Whether to consider periodicity of the structure.
    """
    
    # Map all sites to the unit cell; 0 ≤ xyz < 1
    structure = Structure.from_sites(temp_structure, to_unit_cell=True)

    # Get Voronoi nodes in primitive structure and then map back to the
    # supercell
    prim_structure = structure.get_primitive_structure()

    # Get all atom coords in a supercell of the structure because
    # Voronoi polyhedra can extend beyond the standard unit cell
    coords = []
    if periodicity: cell_range = list(range(-1, 2))  # Periodicity
    else:           cell_range = [0]  # No periodicity
    for shift in itertools.product(cell_range, cell_range, cell_range):
        for site in prim_structure.sites:
            shifted = site.frac_coords + shift
            coords.append(prim_structure.lattice.get_cartesian_coords(shifted))

    # Voronoi tessellation
    voro = Voronoi(coords)

    tol = 1e-6
    new_ridge_points = []
    for atoms in voro.ridge_points:  # Atoms are indexes referred to coords
        # Dictionary for storing information about each atom
        atoms_info = {}

        new_atoms = []
        # Check if any of those atoms belong to the unitcell
        for atom_idx in range(2):
            atom = atoms[atom_idx]

            # Direct coordinates from supercell referenced to the primitive cell
            frac_coords = prim_structure.lattice.get_fractional_coords(coords[atom])

            is_atom_inside = True
            frac_coords_uc = frac_coords
            if not np.all([-tol <= coord < 1 + tol for coord in frac_coords]):
                # atom_x is not inside
                is_atom_inside = False

                # Apply periodic boundary conditions
                while np.any(frac_coords_uc > 1): frac_coords_uc[np.where(frac_coords_uc > 1)] -= 1
                while np.any(frac_coords_uc < 0): frac_coords_uc[np.where(frac_coords_uc < 0)] += 1

            # Obtain mapping to index in unit cell
            uc_idx = np.argmin(np.linalg.norm(prim_structure.frac_coords - frac_coords_uc, axis=1))
            
            if is_atom_inside:
                new_atoms.append(str(uc_idx))
            else:
                new_atoms.append('-'+str(uc_idx))
        
        distance = np.linalg.norm(coords[atoms[1]] - coords[atoms[0]])
        new_atoms.append(distance)
        new_atoms.append(atoms[0])
        new_atoms.append(atoms[1])
        
        new_ridge_points.append(new_atoms)
    
    # Delete those edges which only contain images
    to_delete = []
    for k in range(len(new_ridge_points)):
        pair = new_ridge_points[k][:2]
        if (pair[0][0] == '-') and (pair[1][0] == '-'):
            to_delete.append(k)
    new_ridge_points = np.delete(new_ridge_points, to_delete, axis=0)
    
    edges      = []
    attributes = []
    for idx_i in range(temp_structure.num_sites):
        for idx_j in np.arange(idx_i+1, temp_structure.num_sites):
            to_delete = []
            for k in range(len(new_ridge_points)):
                pair = new_ridge_points[k, :2]
                dist = new_ridge_points[k, 2]
                
                if np.any(pair == str(idx_i)):  # Real for idx_i
                    if pair[0][0] == '-': pair[0] = pair[0][1:]
                    if pair[1][0] == '-': pair[1] = pair[1][1:]
                    
                    if np.any(pair == str(idx_j)):  # Real or image for idx_j
                        edges.append(np.array(pair, dtype=int))
                        attributes.append(float(dist))
                        to_delete.append(k)

            # Delete these added edges, which are no longed needed
            new_ridge_points = np.delete(new_ridge_points, to_delete, axis=0)

    edges      = np.array(edges)
    attributes = np.array(attributes)

    # Generate nodes from all atoms in structure
    nodes = []
    for idx in range(structure.num_sites):
        # Get species type
        species_name = str(structure[idx].species)[:-1]

        # Get node info
        # Loading the node (mass, charge, electronegativity and ionization energy)
        nodes.append([atomic_data[species_name]['atomic_mass'],
                      atomic_data[species_name]['charge'],
                      atomic_data[species_name]['electronegativity'],
                      atomic_data[species_name]['ionization_energy']])
    return nodes, edges, attributes

def get_sphere_images_tessellation(
        atomic_data,
        structure,
        distance_threshold=6,
        solid_solution_data=None
):
    """Gets the distances by pairs of particles, considering images with periodic boundary conditions (PBC).

    Args:
        atomic_data        (dict):                      A dictionary with all node features.
        structure          (pymatgen Structure object): Structure from which the graph is to be generated
        distance_threshold (float, optional):           The distance threshold for edge creation (default is 6).

    Returns:
        nodes      (list): A tensor containing node attributes.
        edges      (list): A tensor containing edge indices.
        attributes (list): A tensor containing edge attributes (distances).
    """

    # structure.get_all_neighbors returns a list of neighbor lists per site
    neighbors = structure.get_all_neighbors(distance_threshold)

    # Adding nodes and edges.
    nodes = []
    edges = []
    attributes = []
    for i, site in enumerate(structure.sites):
        if solid_solution_data is None:
            atomic_mass       = atomic_data[site.species_string]['atomic_mass']
            charge            = atomic_data[site.species_string]['charge']
            electronegativity = atomic_data[site.species_string]['electronegativity']
            ionization_energy = atomic_data[site.species_string]['ionization_energy']
        else:
            atomic_mass = sum(
                solid_solution_data[site.species_string][ss_name] * atomic_data[ss_name]['atomic_mass']
                for ss_name in solid_solution_data[site.species_string].keys())
            charge = sum(
                solid_solution_data[site.species_string][ss_name] * atomic_data[ss_name]['charge']
                for ss_name in solid_solution_data[site.species_string].keys())
            electronegativity = sum(
                solid_solution_data[site.species_string][ss_name] * atomic_data[ss_name]['electronegativity']
                for ss_name in solid_solution_data[site.species_string].keys())
            ionization_energy = sum(
                solid_solution_data[site.species_string][ss_name] * atomic_data[ss_name]['ionization_energy']
                for ss_name in solid_solution_data[site.species_string].keys())

        # Adding the nodes (mass, charge, electronegativity and ionization energies)
        nodes.append([atomic_mass,
                      charge,
                      electronegativity,
                      ionization_energy])

        for neighbor in neighbors[i]:
            j = neighbor.index
            distance = neighbor.nn_distance

            if neighbor.nn_distance > 0:
                # Append edge i->j and j->i to make it undirected
                edges.append([i, j])
                attributes.append([distance])
                if i != j:
                    edges.append([j, i])
                    attributes.append([distance])
    return nodes, edges, attributes

def get_molecule_tessellation(
        atomic_data,
        smiles
):
    """Extracts graph information from SMILES codification of a molecule.

    Args:
        atomic_data (dict): A dictionary with all node features.
        smiles      (str): SMILES string codifying a molecule.

    Returns:
        nodes      (list): A tensor containing node attributes.
        edges      (list): A tensor containing edge indices.
        attributes (list): A tensor containing edge attributes (distances).
    """

    # Generate the molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)

    # Get edge attributes (bond types)
    edges      = []
    attributes = []
    for i in range(mol.GetNumAtoms()):
        for j in range(i+1, mol.GetNumAtoms()):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bond_type = bond.GetBondTypeAsDouble()
                edges.append([i, j])
                attributes.append(bond_type)

    # Generate node features
    nodes = []
    for atom in mol.GetAtoms():
        species_name = atom.GetSymbol()
        nodes.append([atomic_data[species_name]['atomic_mass'],
                      atomic_data[species_name]['charge'],
                      atomic_data[species_name]['electronegativity'],
                      atomic_data[species_name]['ionization_energy']])
    return nodes, edges, attributes


def graph_POSCAR_encoding(
        structure,
        encoding_type='sphere-images',
        distance_threshold=6,
        periodicity=True
):
    """Generates a graph parameters from a POSCAR.
    There are the following implementations:
        1. Voronoi tessellation.
        2. All particles inside a sphere of radius distance_threshold.
        3. Filled space given a cubic box of dimension [0-Lx, 0-Ly, 0-Lz] considering all necessary images.
           It links every particle with the rest for the given set of nodes and edges.

    Args:
        structure          (pymatgen Structure object): Structure from which the graph is to be generated.
        encoding_type      (str):    Framework used for encoding the structure.
        distance_threshold (float):  Distance threshold for sphere-images tessellation.
        periodicity        (bool):   Whether or not to consider periodicity of the structure.
    Returns:
        nodes      (torch tensor): Generated nodes with corresponding features.
        edges      (torch tensor): Generated connections between nodes.
        attributes (torch tensor): Corresponding weights of the generated connections.
    """

    # Loading dictionary of atomic masses
    atomic_data = {}
    with open('input/atomic_masses.dat', 'r') as atomic_data_file:
        for line in atomic_data_file:
            key, atomic_mass, charge, electronegativity, ionization_energy = line.split()
            atomic_data[key] = {
                'atomic_mass':       float(atomic_mass) if atomic_mass != 'None' else None,
                'charge':            int(charge) if charge != 'None' else None,
                'electronegativity': float(electronegativity) if electronegativity != 'None' else None,
                'ionization_energy': float(ionization_energy) if ionization_energy != 'None' else None
            }

    if encoding_type == 'voronoi':
        # Get edges and attributes for the corresponding tessellation
        nodes, edges, attributes = get_voronoi_tessellation(atomic_data,
                                                            structure,
                                                            periodicity)

    elif encoding_type == 'sphere-images':
        # Get edges and attributes for the corresponding tessellation
        nodes, edges, attributes = get_sphere_images_tessellation(atomic_data,
                                                                  structure,
                                                                  distance_threshold=distance_threshold)

    elif encoding_type == 'all-linked':
        # Get edges and attributes for the corresponding tessellation
        nodes, edges, attributes = get_all_linked_tessellation(atomic_data,
                                                               structure)

    elif encoding_type == 'molecule':
        # Get edges and attributes for the corresponding tessellation
        nodes, edges, attributes = get_molecule_tessellation(atomic_data,
                                                             structure)

    else:
        sys.exit('Error: encoding type not available.')

    # Convert to torch tensors and return
    nodes      = torch.tensor(nodes,      dtype=torch.float)
    edges      = torch.tensor(edges,      dtype=torch.long)
    attributes = torch.tensor(attributes, dtype=torch.float)
    return nodes, edges, attributes