import numpy as np
import torch
import itertools
import sys

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

        if solid_solution_data is None:
            atomic_mass       = atomic_data[species_name]['atomic_mass']
            charge            = atomic_data[species_name]['charge']
            electronegativity = atomic_data[species_name]['electronegativity']
            ionization_energy = atomic_data[species_name]['ionization_energy']
        else:
            atomic_mass       = sum(solid_solution_data[species_name][ss_name] * atomic_data[ss_name]['atomic_mass']
                                    for ss_name in solid_solution_data[species_name].keys())
            charge            = sum(solid_solution_data[species_name][ss_name] * atomic_data[ss_name]['charge']
                                    for ss_name in solid_solution_data[species_name].keys())
            electronegativity = sum(solid_solution_data[species_name][ss_name] * atomic_data[ss_name]['electronegativity']
                                    for ss_name in solid_solution_data[species_name].keys())
            ionization_energy = sum(solid_solution_data[species_name][ss_name] * atomic_data[ss_name]['ionization_energy']
                                    for ss_name in solid_solution_data[species_name].keys())
        
        # Adding the nodes (mass, charge, electronegativity and ionization energies)
        nodes.append([atomic_mass,
                      charge,
                      electronegativity,
                      ionization_energy])    

        # Get the initial position
        position_0 = positions[index_0]
        position_cartesian_0 = np.dot(position_0, structure.lattice.matrix)

        # Explore images of all particles in the system
        # Starting on index_0, thus exploring possible images with itself (except for i,j,k=0, exact same particle)
        for index_i in np.arange(index_0, total_particles):
            # Get the initial position
            position_i = positions[index_i]

            reference_distance_i = np.nan  # So it outputs False when first compared with another distance
            i = 0
            alpha_i = 1
            while True:
                minimum_distance_i   = np.nan
                reference_distance_j = np.nan
                j = 0
                alpha_j = 1
                while True:
                    minimum_distance_j   = np.nan
                    reference_distance_k = np.nan
                    k = 0
                    alpha_k = 1
                    while True:
                        # Move to the corresponding image and convert to cartesian distances
                        position_cartesian_i = np.dot(position_i + [i, j, k], structure.lattice.matrix)

                        # New distance as Euclidean distance between both reference and new image particle
                        new_distance = np.linalg.norm([position_cartesian_0 - position_cartesian_i])

                        # Condition that remove exact same particle
                        same_index_condition     = (index_0 == index_i)
                        all_index_null_condition = np.all([i, j, k] == [0]*3)
                        same_particle_condition  = (same_index_condition and all_index_null_condition)

                        # Applying threshold to images
                        if (new_distance <= distance_threshold) and not same_particle_condition:
                            # Append this point as a edge connection to particle 0
                            edges.append([index_0, index_i])
                            attributes.append([new_distance])

                        # Change direction or update i,j if the box is far
                        elif new_distance > reference_distance_k:
                            # Explore other direction or cancel
                            if alpha_k == 1:
                                k = 0
                                alpha_k = -1
                            else:
                                break

                        reference_distance_k = new_distance
                        k += alpha_k

                        if not minimum_distance_j <= reference_distance_k:
                            minimum_distance_j = reference_distance_k

                    # If k worked fine, j is fine as well thus continue; else, explore other direction or cancel
                    if minimum_distance_j > reference_distance_j:
                        if alpha_j == 1:
                            j = 0
                            alpha_j = -1
                        else:
                            break

                    # Update j
                    j += alpha_j
                    reference_distance_j = minimum_distance_j

                    if not minimum_distance_i <= reference_distance_j:
                        minimum_distance_i = reference_distance_j

                # If j did not work fine, explore other direction or cancel
                if minimum_distance_i > reference_distance_i:
                    if alpha_i == 1:
                        i = 0
                        alpha_i = -1
                    else:
                        break

                # Update i
                i += alpha_i
                reference_distance_i = minimum_distance_i
    return nodes, edges, attributes


def graph_POSCAR_encoding(
        path_to_structure,
        distance_threshold=6,
        solid_solution_data=None
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
    structure = Structure.from_file(path_to_structure)
    
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

