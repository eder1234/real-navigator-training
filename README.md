# RPI Visual Navigation -- RGB‑D + LightGlue

This repository contains code and a reference notebook for **learning
navigation actions from RGB‑D image pairs** collected along real robot
trajectories.\
The pipeline extracts geometric and appearance features between a
**current observation** and a **stored visual memory keyframe**, then
trains a classifier to predict navigation actions.

The project was designed for **real‑world visual navigation
experiments** using RGB‑D sensors and feature matching (LightGlue).

------------------------------------------------------------------------

# Overview

The system models navigation as a **supervised learning problem**.\
For each time step along a trajectory:

1.  A **current RGB‑D frame** is captured.
2.  A **keyframe from visual memory** is selected.
3.  The system computes **feature relationships** between the two
    frames:
    -   RGB‑D similarity
    -   feature matches
    -   relative pose estimation
4.  These features are used to **train a classifier** that predicts the
    correct action.

Typical navigation actions include:

-   forward
-   left
-   right
-   update_memory

The notebook trains and evaluates a model to learn these actions from
recorded trajectories.

------------------------------------------------------------------------

# Main Components

## 1. Visual Memory

Each trajectory contains a **visual memory** composed of selected
keyframes.

    visual_memory-1stdev/

These keyframes represent the path that the robot should follow.

------------------------------------------------------------------------

## 2. RGB‑D Similarity

Implemented in:

    modules/rgbd_similarity.py

This module computes similarity between two RGB‑D images.

Typical signals extracted:

-   RGB appearance similarity
-   depth similarity
-   combined RGB‑D similarity score

These features help determine whether the robot is close to a known
visual location.

------------------------------------------------------------------------

## 3. Feature‑Based Point Cloud Registration

Implemented in:

    modules/feature_based_point_cloud_registration.py

Steps:

1.  Detect keypoints in RGB images
2.  Match features using **LightGlue**
3.  Back‑project matches using depth
4.  Build two point clouds
5.  Estimate **rigid transformation** using SVD

Outputs include:

-   translation magnitude
-   rotation (quaternion)
-   RMSE registration error
-   number of matched points

These provide **geometric cues for navigation decisions**.

------------------------------------------------------------------------

# Machine Learning Pipeline

The notebook:

    real_nav_lightglue_rgbd_notebook_revised.ipynb

performs the full pipeline.

### Step 1 --- Data extraction

For each trajectory:

-   read `Log_Robot.csv`
-   ignore `Idle` actions
-   pair frames with visual memory keyframes
-   compute features

Output:

    features_full_partial.csv
    features_full.csv

------------------------------------------------------------------------

### Step 2 --- Training dataset creation

Filter usable samples and build the dataset:

    features_trainable.csv

Features include:

-   RGB similarity
-   depth similarity
-   LightGlue match statistics
-   relative pose features

------------------------------------------------------------------------

### Step 3 --- Train classifier

The notebook trains:

    MLPClassifier (scikit‑learn)

Preprocessing:

-   StandardScaler
-   LabelEncoder

Artifacts saved:

    scaler.joblib
    label_encoder.joblib
    mlp_lightglue_rgbd_real.joblib

------------------------------------------------------------------------

### Step 4 --- Evaluation

The model is evaluated on a **test split**.

Metrics:

-   balanced accuracy
-   macro F1
-   weighted F1
-   confusion matrix

Predictions are exported:

    test_predictions.csv

------------------------------------------------------------------------

# Repository Structure

Expected structure:

    rpi_nav/
    │
    ├── real_nav_lightglue_rgbd_notebook_revised.ipynb
    │
    ├── modules/
    │   ├── rgbd_similarity.py
    │   ├── feature_based_point_cloud_registration.py
    │   └── feature_matcher.py
    │
    ├── visual_paths/
    │   ├── Almacen-Laboratorio_J/
    │   │   ├── rgb/
    │   │   ├── depth/
    │   │   ├── Log_Robot.csv
    │   │   ├── Simulation/
    │   │   └── visual_memory-1stdev/
    │   │
    │   ├── Cuarto_O-Laboratorio_M/
    │   ├── Cubiculo_3-Laboratorio_E/
    │   └── Sala_E-Laboratorio_A/
    │
    ├── notebook_outputs/
    │   └── real_nav_lightglue/
    │       ├── features_full.csv
    │       ├── features_full_partial.csv
    │       ├── features_trainable.csv
    │       ├── scaler.joblib
    │       ├── label_encoder.joblib
    │       ├── mlp_lightglue_rgbd_real.joblib
    │       └── test_predictions.csv
    │
    └── lightglue/
        └── feature matching implementation

------------------------------------------------------------------------

# Installation

Recommended: **Python 3.9+**

Install dependencies:

``` bash
pip install numpy
pip install pandas
pip install scikit-learn
pip install torch
pip install opencv-python
pip install matplotlib
pip install joblib
```

LightGlue dependencies:

``` bash
pip install kornia
pip install einops
```

If using CUDA for feature matching:

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

------------------------------------------------------------------------

# Running the Pipeline

Open the notebook:

    real_nav_lightglue_rgbd_notebook_revised.ipynb

and execute all cells.

Outputs will be stored in:

    notebook_outputs/real_nav_lightglue/

------------------------------------------------------------------------

# Expected structure
```
visual_paths/
├── Almacen-Laboratorio_J/
│   ├── Log_Robot.csv
│   ├── Simulation/
│   │   ├── rgb/
│   │   │   ├── rgb_0000.png
│   │   │   ├── rgb_0001.png
│   │   │   └── ...
│   │   └── depth/
│   │       ├── depth_0000.png
│   │       ├── depth_0001.png
│   │       └── ...
│   └── visual_memory-1stdev/
│       ├── rgb/
│       │   ├── rgb_0000.png
│       │   ├── rgb_0001.png
│       │   └── ...
│       └── depth/
│           ├── depth_0000.png
│           ├── depth_0001.png
│           └── ...
├── Cuarto_O-Laboratorio_M/
│   ├── Log_Robot.csv
│   ├── Simulation/
│   │   ├── rgb/
│   │   │   ├── rgb_0000.png
│   │   │   └── ...
│   │   └── depth/
│   │       ├── depth_0000.png
│   │       └── ...
│   └── visual_memory-1stdev/
│       ├── rgb/
│       │   ├── rgb_0000.png
│       │   └── ...
│       └── depth/
│           ├── depth_0000.png
│           └── ...
├── Cubiculo_13-Laboratorio_I/
│   ├── Log_Robot.csv
│   ├── Simulation/
│   │   ├── rgb/
│   │   │   ├── rgb_0000.png
│   │   │   └── ...
│   │   └── depth/
│   │       ├── depth_0000.png
│   │       └── ...
│   └── visual_memory-1stdev/
│       ├── rgb/
│       │   ├── rgb_0000.png
│       │   └── ...
│       └── depth/
│           ├── depth_0000.png
│           └── ...
├── Cubiculo_3-Laboratorio_E/
│   ├── Log_Robot.csv
│   ├── Simulation/
│   │   ├── rgb/
│   │   │   ├── rgb_0000.png
│   │   │   └── ...
│   │   └── depth/
│   │       ├── depth_0000.png
│   │       └── ...
│   └── visual_memory-1stdev/
│       ├── rgb/
│       │   ├── rgb_0000.png
│       │   └── ...
│       └── depth/
│           ├── depth_0000.png
│           └── ...
└── Sala_E-Laboratorio_A/
    ├── Log_Robot.csv
    ├── Simulation/
    │   ├── rgb/
    │   │   ├── rgb_0000.png
    │   │   └── ...
    │   └── depth/
    │       ├── depth_0000.png
    │       └── ...
    └── visual_memory-1stdev/
        ├── rgb/
        │   ├── rgb_0000.png
        │   └── ...
        └── depth/
            ├── depth_0000.png
            └── ...
```
------------------------------------------------------------------------

# Notes

Important assumptions:

-   RGB and depth filenames are synchronized
-   `Log_Robot.csv` contains the navigation actions
-   trajectories are stored in `visual_paths/`

The notebook also performs **data quality auditing** before feature
extraction.

------------------------------------------------------------------------
