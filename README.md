# CS543 Project: Shelf Management

This repository contains the codebase for our CS543 project on **Shelf Management: A Deep Learning-Based System for Shelf Visual Monitoring**. It implements an end-to-end pipeline for detecting shelf rows and recognizing products to assess planogram compliance in retail environments.

## Project Structure

- `scripts/`: Contains Python scripts for training, testing, and visualizing the recognition pipeline, as well as the Jupyter Notebooks for the end-to-end evaluation (`Product_Recognition_Pipeline.ipynb` and `Shelf_Row_Detection_Pipeline.ipynb`).
- `result/`: Contains output visualizations and results (e.g., `visualization_results.png`).

## Setup Instructions

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your dataset. The expected format includes `training_set` and `test_set` directories inside the `data/` folder, which should be symlinked to the project root.

## Acknowledgements

This project builds upon the work in the paper [Shelf Management: a Deep Learning-Based system for shelf visual monitoring](https://doi.org/10.1016/j.eswa.2024.124635). The original dataset and detection methodologies have been adapted for our class project.
