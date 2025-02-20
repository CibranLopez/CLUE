# CLUE: Comparative Latent Uncertainty Estimator

Doctors often diagnose by comparing patients to similar past cases, focusing on relevant attributes like symptoms or test results. This selective reasoning mirrors how machine learning models use latent spaces to make predictions, emphasizing task-specific features while ignoring irrelevant ones.

We introduce CLUE, a novel uncertainty quantification (UQ) approach leveraging deep learning encodings. Our framework efficiently estimates uncertainty by comparing the latent representation of unseen samples to those in the training set, capturing both chemical and geometric structure as well as environmental constraints. It provides reliable confidence metrics, flagging predictions that arise from extrapolation or lack sufficient similarity to known data.

Please be aware that the code is under active development, bug reports are welcomed in the GitHub issues!

## Installation

To download the repository and install the dependencies:

```bash
git clone https://github.com/CibranLopez/CLUE.git
cd CLUE
pip3 install -r requirements.txt
```

## Execution

A set of user-friendly jupyter notebook canm be found on the examples section, which can be run locally. These models train a GCNN and make predictions with uncertainty estimations (recall that CLUE can be used in any already-trained model, even with different technologies from a GNN).

## Authors

This project is being developed by:

- **Cibrán López Álvarez** - Lead Developer and Researcher

## Contact, questions and contributing

For any questions, issues, or contributions, feel free to contact:

- Cibrán López Álvarez: [cibran.lopez@upc.edu](mailto:cibran.lopez@upc.edu)

Feel free to open issues or submit pull requests for bug fixes, improvements, or feature suggestions.
