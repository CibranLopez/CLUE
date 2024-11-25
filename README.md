# CLUE: Comparative Latent Uncertainty Estimator

This project introduces a novel framework for generating new crystal materials using equivariant diffusion models based on a graph-like representation of data. This offers a powerful approach to surpass previous implementations, as it considers interactions and similarities between close points (which are more likely to interact). As well, aligned with previous implementations, ours also allows maximizing arbitrary targets such as conductivity, absorption, and more, or looking for materials with specific properties (such as diffraction patterns or n-order transitions). The generated crystals demonstrate remarkable thermodynamic stability (convex-hull approach), compared against the Materials Project database. 

The core technology behind this framework is based on deep convolutional layers and graph-like representation of data, where the diffusion process is achieved through the use of Markov chains. The denoising aspect of the model is implemented using convolutional graph neural networks, ensuring high-quality results, with which the noise of graphs is predicted and extracted, allowing the generation of an arbitrary number of novel, independent materials.

This technology is suitable for different applications: from discovering improved ionic conductors beyond current databases to generating molecules for efficient water splitting. Moreover, the model itself can be applied to a variety of problems (concretely, any problem which can be formulated in terms of graphs), such as proposing enhanced distributions in social networks or traffic. Then, although applied to crystal material generation, this repository is divided into two independent functionalities:

The main feature is the generation and interpolation of novel graphs given a reference database of them. However, this architecture can be directly applied to any other problem whose data is graph-like.

An extended discussion on all these topics can be found in our [paper](https://www.overleaf.com/read/cjxhknmhpfpg#d4cb5f).

## Features

This repository shares the new CLUE technologies with comparison against two state-of-the-art approaches.

Please be aware that the code is under active development, bug reports are welcomed in the GitHub issues!

## Installation

To download the repository and install the dependencies:

```bash
git clone https://github.com/CibranLopez/CLUE.git
cd CLUE
pip3 install -r requirements.txt
```

## Execution

A set of user-friendly jupyter notebook have been developed, which can be run locally with pytorch dependencies. It makes predictions on a target dataset and evaluates uncertainty via CLUE, bayesian models and ensembles.

## Authors

This project is being developed by:

- **Cibrán López Álvarez** - Lead Developer and Researcher

## Contact, questions and contributing

For any questions, issues, or contributions, feel free to contact:

- Cibrán López Álvarez: [cibran.lopez@upc.edu](mailto:cibran.lopez@upc.edu)

Feel free to open issues or submit pull requests for bug fixes, improvements, or feature suggestions.
