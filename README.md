# **Training Models**
## **Overview** 
### Data Processing Pipeline

![Pipeline](https://github.com/RumiaKitinari/Enhancing-Skin-Disease-Classification-with-Transformer-Architecture-and-GLIDE-Stable-Diffusion-Model/blob/main/Our_Model_Train_and_Visuals/model_tables_visuals/pipeline.png)

1. **Data Loading:** Load the specified dataset using Hugging Face datasets
2. **Preprocessing:** 
	1. Applies transforms (random resized crop, horizontal flip, normalization)
	2. (Technically at this stage) training GLIDE to apply pre-processing to synthesized images
3. **Model Loading:** Loads pre-trained model with custom classification head
4. **Training:** Fine-tunes for 10 epochs with configurable batch size and learning rate (5e-5)
5. **Evaluation:** Computes accuracy on validation set after each epoch
6. **Testing:** Evaluates final model on held-out test set
7. **Results:** Save model locally

### Datasets

| Dataset | Description |
|---------|-------------|
| `HAM10000` | Original HAM10000 dataset |
| `Atlas + ISIC` | Combined Atlas and ISIC dataset |
| `GLIDE HAM10000` | HAM10000 augmented with GLIDE (+60 samples per class) |
| `GLIDE Atlas + ISIC` | Atlas + ISIC augmented with GLIDE (+60 samples per class) |

### Models Evaluated

1. **DinoV2**
2. **Swin** (Swin Transformer)
3. **Vision Transformer** (ViT)


| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Learning Rate | 5e-5 |
| Warmup Ratio | 0.1 |
| Gradient Accumulation Steps | 4 |
| Evaluation Strategy | Per epoch |
| Save Strategy | Per epoch |
| Metric for Best Model | Accuracy |
| Batch | Model Specific |



## **STEP 0:** Downloads
The notebook automatically installs:
- `torch` & `torchvision`
- `transformers`, `datasets`, `accelerate`, `evaluate`
- `scikit-learn`, `pillow`, `ipywidgets`, `ipykernel`

You must manually install:
1. `glide_text2im`
2. `os`, `sys`
3. `pandas`, `shuntil`, `sklearn`, `huggingface_hub`, `datasets`

You can also install:
1. `IPython.display`

## **STEP 1:** Downloading Datasets

1. *STEP 1.1:* Clone or otherwise import this Repository. 
2. *STEP 1.2:* Go to `file_management_scripts`
3. *STEP 1.3:* For each of the following scripts, edit the `target_folder` to a folder “datasets”. Note that all other downloaded datasets should be stored in the same location. 
	1. `ham10000_download.py`
	2. `Atlas_ISIC_download.py`
4. *STEP 1.4:* Execute both of these files.
	1. *Optional Instructions:* Enter a terminal. Then, use `cd` to navigate to the directory which contains `file_management_scripts`. Lastly, run the commands below:
	2. `python -m ham_10000_download`
	3. `python -m Atlas_ISIC_download`
5. *STEP 1.5:* For the script `Atlas_ISIC_merge.py`, edit `base_path`, `train_dir`,  `test_dir`, and the merged-directory to the “datasets” file path from `STEP 1.3`. 
6. *STEP 1.6:* For each of the following scripts…
	1. Edit the `data_dir` and `output_dir` to be the file paths from `STEP 1.3`.
	2. Execute both of these files. 
		1. *Optional Instructions:* Run the commands below:
		2. `python -m ham10000_split`
		3. `python -m Atlas_ISIC_download`
7. *STEP 1.7:* Navigate to `ham10000_supersplit`. 
	1. Edit `data_dir` and `output_dir` to be the file path from `STEP 1.3`. 
	2. Execute the file. Refer to previous *Optional Instructions* as needed.
8. *STEP 1.8:* Navigate to `glide_split.py`.
	1. Edit the `base_path` to be the file path from `STEP 1.3`. 
	2. Execute the file. Refer to previous *Optional Instructions* as needed.

## **STEP 2:** Training Model on GLIDE
1. *STEP 2.1:* Navigate to the file `GLIDE Synthesis.ipynb` in folder `GLIDE_Synthesis`. Open the file. 
2. *STEP 2.2:* Navigate to “STEP 2: Executing Model Training for the Dataset”. Below that is a block `train_ds = load_dataset(...)`. 
3. *STEP 2.3:* Edit the `data_dir` to be the combination of the file path from `STEP 1.3` and `/GLIDE_split`. 
4. *(OPTIONAL) STEP 2.4:* Consider uncommenting code block 11 (`th.save(...)`) and editing the file path to one of your choice to save the updated model parameters.
	1. *NOTE:* Consider uncommenting code block 12 (`base path = ...`) to retrieve that save for future runs.
5. *STEP 2.5:* Navigate to code block 23 (`names = train_dataloader.dataset...`) and edit the `output_dir` on line 16 to the file path from `STEP 1.3`. 
6. *STEP 2.6:* Execute the file. Refer to previous *Optional Instructions* as needed. 
	1. *WARNING:* This step may take upwards of 12 hours. Consider executing the `.py` version of this code in the terminal of a server. Update the code in `GLIDE synthesis.py` with instructions `STEP 2.2` - `STEP 2.5`. 
7. *STEP 2.7:* Go to `file_management_scripts`
8. *STEP 2.8:* For `Atlas_ISIC_GLIDE_merge` and `HAM_GLIDE_merge`, follow these instructions:
	1. Edit the `reg_dir` / `reg_dir` / `merged_dir` to the file path from `STEP 1.3`. To be specific, preserve the elements after `/datasets/`, and update the path before and including `/datasets/` with the file path form `STEP 1.3`. 
	2. Execute the file. Refer to previous *Optional Instructions* as needed.

## **STEP 3:** Executing Model Training
### Configurable Parameters
At the top of the notebook, you can configure the following parameters:
1. **(Optional) GPU Core Allocation:** Selecting 2 or 3 for our project. 
2. **Model Selection:** Choose from three pre-trained vision models:
	1. "swin" - Microsoft Swin Transformer (batch size: 16)
	2. "dino" - Facebook DINOv2 Base (batch size: 32)
	3. "vit" - Google ViT Base Patch16-224 (batch size: 32)
3. **Dataset Selection:** Choose from two base datasets:
	1. "ham" - HAM10000 dataset (7 skin disease classes)
	2. "atlas_isic" - Atlas+ISIC dataset
4. **GLIDE Augmentation:**
	1. glide_choice = True - Use GLIDE-augmented dataset (synthetic images)
	2. glide_choice = False - Use original dataset

### Total Configuration Combinations
This notebook supports 12 unique training configurations. All 12 have been successfully produced and run:

| # | Model | Dataset | GLIDE |
|---|-------|---------|-------|
| 1 | Swin | HAM10000 | False |
| 2 | Swin | HAM10000 | True |
| 3 | Swin | Atlas+ISIC | False |
| 4 | Swin | Atlas+ISIC | True |
| 5 | DINO | HAM10000 | False |
| 6 | DINO | HAM10000 | True |
| 7 | DINO | Atlas+ISIC | False |
| 8 | DINO | Atlas+ISIC | True |
| 9 | ViT | HAM10000 | False |
| 10 | ViT | HAM10000 | True |
| 11 | ViT | Atlas+ISIC | False |
| 12 | ViT | Atlas+ISIC | True |

### Instructions
To execute and acquire results from all models and all datasets, refer to the parameter and configurations specified. Execute all 12 unique training configurations to recreate the model results. 

*NOTE:* See `Visualizing Models` below for options for generating plots or otherwise visualizing the data. 



# **Visualizing Models**
# `models/`
This folder contains all model training notebooks for the project. Multiple copies of similar notebooks exist to enable parallel training runs across different configurations, reducing total training time. Atlhough we have provided a master notebook with comments and explainations of code. 

### File Organization
`model.ipynb` is the file in which we have been able to run every model. The few changes which we have made are 
- **`{model}.ipynb`** – Main training notebook for a model on raw (non-augmented) data
- **`{model}_glide_(dataset).ipynb`** – Model training on GLIDE-augmented datasets (code modified to handle augmented data structure)
- **`model_saves/`** – Subdirectory containing saved model checkpoints from each training run (used by `model_performance_metrics` and `model_tables_visuals`)


## `class_distribution_plots/`

Generates class distribution bar charts for our datasets.

- **`dataset_dist_barchart.ipynb`** :Main notebook for generating plots
	- Code was ChatGPT-generated then manually refined
	- GLIDE-augmented dataset excluded (fixed +60 per class would mask true distribution)
	- Generation of cluster image to represent sample images from the datasets

- Visualizing dataset balance in our research project.
- Visualizing sample images from respective datasets.

## `model_performance_metrics/`

This folder contains model evaluation results and performance tracking across different training runs.

### Contents

- **`model_performance_metrics.ipynb`** – Main evaluation notebook (primarily ChatGPT-generated code)
	- Loads every saved model state from training
	- Tests accuracy on validation/test sets
	- Exports raw, unformatted results to `model_evaluation_full_results.csv`


- **`model_evaluation_full_results.csv`** – Raw, unformatted evaluation data for all model checkpoints


- **`table_full_formatted.csv`** – Clean, formatted version of the full evaluation results
	- Generated by `Aiden-Enhanced/model_tables_visuals/model_tables_charts.ipynb`
	- Stored here for convenient access and sorting
  

### Legacy Files (from early presentation)

- **`model_performance_NoGLIDE.csv`** – Early results before GLIDE training
- **`model_performance_table_NoGLIDE.html`** – HTML table for presentation (pre-GLIDE)
  

> **Note:** The legacy files are preserved for reference but are no longer actively used, as the project has moved on to include GLIDE-augmented training.

## `model_tables_visuals/`

Publication-ready visualizations and tables for the final paper.

- **`model_tables_visuals.ipynb`** – Main notebook (ChatGPT-generated + manual refinements)
	- Imports `model_evaluation_full_results.csv` from `../model_performance_metrics/`
	- Generates formatted tables and comparison figures

### Outputs

| File | Description |
|------|-------------|
| `heatmaps_original_vs_glide.png` | Heatmap comparing original vs. GLIDE-augmented dataset performance |
| `radar_chart.png` | Multi-dimensional performance radar chart (precision, recall, F1, accuracy, class-wise metrics) |
| `*.csv` / `*.html` | Formatted results table for publication |

**Purpose:** Clear, accessible visual comparison of GLIDE augmentation impact for the final research paper.

## `model_training_lineplots`

This folder contains the notebook for generating training visualization plots across different model architectures and dataset configurations.

### Contents

- **`model_train_plots.ipynb`** – Main notebook containing hardcoded training history data and visualization functions

### Overview

The notebook generates publication-ready learning curves and accuracy comparison plots for three model architectures trained on four dataset configurations.

### Models Evaluated

- **DinoV2**
- **Swin** (Swin Transformer)
- **Vision Transformer** (ViT)

### Datasets

| Dataset | Description |
|---------|-------------|
| `HAM10000` | Original HAM10000 dataset |
| `Atlas + ISIC` | Combined Atlas and ISIC dataset |
| `GLIDE HAM10000` | HAM10000 augmented with GLIDE (+60 samples per class) |
| `GLIDE Atlas + ISIC` | Atlas + ISIC augmented with GLIDE (+60 samples per class) |

### Generated Outputs

The notebook produces two types of plots saved to the `plots/` directory:

| Plot Type | File Pattern | Description |
|-----------|--------------|-------------|
| Learning curves | `{Model}_{Dataset}_learning_curve.png` | Side-by-side loss and accuracy plots per model-dataset pair |
| Accuracy comparison | `{Dataset}_accuracy_comparison.png` | Overlaid accuracy curves comparing all three models on the same dataset |

### Implementation Notes

- Training history data is **hardcoded** directly in the notebook
- Generated plots are **manually reviewed and sorted** for final selection
- GLIDE-augmented datasets have validation loss marked as `"No log"` (not logged during training)
- Final test accuracy is annotated on accuracy plots with a red marker

### Usage

Run all cells sequentially to generate all plots. Outputs are saved to the `plots/` folder within this directory.

