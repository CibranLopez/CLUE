import numpy as np
import torch
import itertools
import sys
import os
import json

from pymatgen.core.structure import Structure

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            atomic_mass = atomic_data[site.species_string]['atomic_mass']
            charge = atomic_data[site.species_string]['charge']
            electronegativity = atomic_data[site.species_string]['electronegativity']
            ionization_energy = atomic_data[site.species_string]['ionization_energy']
        else:
            atomic_mass = sum(solid_solution_data[species_name][ss_name] * atomic_data[ss_name]['atomic_mass']
                              for ss_name in solid_solution_data[site.species_string].keys())
            charge = sum(solid_solution_data[species_name][ss_name] * atomic_data[ss_name]['charge']
                         for ss_name in solid_solution_data[site.species_string].keys())
            electronegativity = sum(
                solid_solution_data[species_name][ss_name] * atomic_data[ss_name]['electronegativity']
                for ss_name in solid_solution_data[site.species_string].keys())
            ionization_energy = sum(
                solid_solution_data[species_name][ss_name] * atomic_data[ss_name]['ionization_energy']
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

def graph_POSCAR_encoding(
        path_to_structure,
        distance_threshold=6
):
    """Generates a graph parameters from a POSCAR.

    Args:
        path_to_structure  (str):   Path to structure file from which the graph is generated.
        distance_threshold (float): Distance threshold for sphere-images tessellation.
    Returns:
        nodes      (torch tensor): Generated nodes with corresponding features.
        edges      (torch tensor): Generated connections between nodes.
        attributes (torch tensor): Corresponding weights of the generated connections.
    """

    # Load pymatgen structure object
    structure = Structure.from_file(f'{path_to_structure}/POSCAR')
    
    # Loading dictionary of atomic masses
    atomic_data = {}
    with open('input/atomic_masses.dat', 'r') as atomic_data_file:
        for line in atomic_data_file:
            key, atomic_mass, charge, electronegativity, ionization_energy = line.split()
            atomic_data[key] = {
                'atomic_mass':       float(atomic_mass)       if atomic_mass       != 'None' else None,
                'charge':            int(charge)              if charge            != 'None' else None,
                'electronegativity': float(electronegativity) if electronegativity != 'None' else None,
                'ionization_energy': float(ionization_energy) if ionization_energy != 'None' else None
            }

    solid_solution_data = None
    if os.path.exists(f'{path_to_structure}/solid-solution.json'):
        # Load the JSON file
        with open(f'{path_to_structure}/solid-solution.json', 'r') as json_file:
            solid_solution_data = json.load(json_file)
    
    # Get edges and attributes for the corresponding tessellation
    nodes, edges, attributes = get_sphere_images_tessellation(atomic_data,
                                                              structure,
                                                              distance_threshold=distance_threshold,
                                                              solid_solution_data=solid_solution_data)

    # Convert to torch tensors and return
    nodes      = torch.tensor(nodes,      dtype=torch.float)
    edges      = torch.tensor(edges,      dtype=torch.long)
    attributes = torch.tensor(attributes, dtype=torch.float)
    return nodes, edges, attributes

