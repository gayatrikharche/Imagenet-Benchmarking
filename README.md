# Imagenet benchmarkng on OSG

# Initial Tries and Learning Path

Training models on the ImageNet dataset presents significant challenges due to the dataset's large size and complexity. To efficiently utilize computational resources on the Open Science Grid (OSG), effective strategies for managing and processing this dataset are essential. The goal is to benchmark various methods for handling the dataset, evaluating their efficiency, resource usage, and overall feasibility.

In this context, three methods were explored for training ImageNet on OSG:

### Method 1: Traditional Download and Extraction with HTCondor Native Handling
This approach involved downloading the dataset directly to the Access Point (AP) using HTCondor’s native file transfer capabilities. The dataset was then extracted locally for model training. 

- **Advantages**: Straightforward method; minimal dependencies on external protocols.
- **Challenges**: Substantial local storage was required, and this method was constrained by HTCondor's file transfer size limits, which capped at 5GB per job. This made it impractical for datasets as large as ImageNet (150GB).

### Method 2: Direct Fetching from the Origin with HTTP
This method simplified the process by fetching the dataset directly from the origin via HTTPS within the job script. The script handled dataset retrieval, extraction, training, and cleanup in a single workflow.

- **Advantages**: Reduced the need for pre-job data management and alleviated storage pressure on the AP.
- **Challenges**: HTTP transfers were not optimized for large datasets like ImageNet, leading to slower download times and occasional transfer failures.

### Method 3: Direct Fetching from the Origin Using OSDF Protocol
This method utilized the OSDF protocol to efficiently fetch the dataset from the SDSC Pelican storage, streamlining data handling and reducing storage needs on the AP. The protocol leveraged OSDF redirectors and caches for better data transfer efficiency.

- **Advantages**: Optimized for large datasets, minimizing transfer bottlenecks and storage constraints on the AP. It also integrated seamlessly with HTCondor and the OSG infrastructure.
- **Challenges**: Initial setup of the OSDF protocol required some configuration effort but yielded significant improvements in resource management.

---

For the main benchmarking run of 100 jobs, **Method 3** (Direct Fetching with OSDF) was used. This method proved to be the most effective, allowing us to handle the large ImageNet dataset efficiently across OSG, while minimizing storage requirements and improving data transfer speeds.


# Executive Summary

We explore the feasibility and effectiveness of running ML workloads on the OSPool, with data handling mediated by OSDF/Pelican. This document focuses on running GPU-accelerated ML training using the 150GB ImageNet dataset, following the recommended OSG method [OSG PyTorch Tutorial](https://github.com/OSGConnect/tutorial-pytorch). The ML training job uses modest resources, with a single GPU per job and about 4 hours of GPU time required to complete.

The ImageNet dataset was hosted on the SDSC OSDF/Pelican origin, with HTCondor retrieving it using the osdf protocol. This protocol uses the OSDF redirector and caches to manage data transfers.

### Key Observations:
- **Job Completion**: It took one week to complete 100 ML training jobs, using 685 GPU hours.
- **Success Rate**: Approximately 80% of the jobs were successful, while 20% failed.
- **Compute Efficiency**: Only 47% of the total wallclock time was spent on successful computing, with 7% spent on file transfers, leading to a 14% file transfer overhead.
- **Waste**: 44% of total time was wasted on retried file transfers, while preemption caused minimal waste. This contributed to longer job completion times.
- **Input File Handling**: OSDF is necessary as OSPool enforces a 5GB per-job input file size limit for native HTCondor transfers.

---

### Total Statistics for 100 ML Training Jobs:

| Description                                                      | Total Time  |
|------------------------------------------------------------------|-------------|
| Number of jobs that got held (no automated recovery)              | 18          |
| Number of jobs that failed due to wrong output                    | 3           |
| Total time spent computing for successful jobs                    | 318.6 hours |
| Total time spent transferring files for successful jobs           | 45.8 hours  |
| Total time wasted due to retried transfers (successful jobs)      | 302.2 hours |
| Total time wasted in compute due to preemption                    | 16 hours    |
| Total time wasted in file transfers for preempted jobs            | 0.8 hours   |
| Total time wasted in compute due to wrong outputs                 | 0.6 hours   |
| Total time wasted in file transfers for wrong outputs             | 2.2 hours   |
| Mean time successful jobs took from matching to completion        | 8.6 hours   |
| Median time successful jobs took from matching to completion      | 4.9 hours   |

---

# Evaluation Details

## Dataset and Script Used for Benchmarking

**ImageNet**: The 150GB ImageNet dataset was selected as a benchmark because of its 1.2 million images across 1,000 categories, testing a model’s ability to generalize. We converted the original zip file into tar format since the OSG container image doesn't support `unzip`.

- **Dataset**: `ILSVRC2012_img_train.tar`
  - **Structure**: 
    - The tar archive contains images organized into class-specific directories.
    - Each class folder includes images belonging to that class.
  - **Size**: 137 GB.

## Python Script Overview

This script trains a ResNet50 model on an ImageNet-like dataset using PyTorch. It includes:

- **Custom Dataset**: The `CustomImageNetDataset` class loads and preprocesses images from a directory structure similar to ImageNet.
- **Transformations**: The dataset undergoes transformations such as resizing, flipping, tensor conversion, and normalization for ResNet50 compatibility.
- **ResNet50 Model**: A pre-trained ResNet50 model is adapted by modifying the final layer for 1,000 ImageNet categories.
- **Training Process**: The model is trained using SGD, and results are logged at every step. After training, the model is saved as `imagenet_resnet50.pt`.

## Main Benchmarking Run

- **Execution Environment**: The jobs were run using `SingularityImage = "osdf:///ospool/uc-shared/public/OSG-Staff/pytorch-2.1.1.sif"`.
- **Data Fetching**: Dataset accessed via `osdf:///nrp/osdf/ILSVRC2012/ILSVRC2012_img_train.tar`.
- **Wrapper Script**: The script manages data, extracts files, and cleans up after training. It trains the ResNet50 model for 5 epochs using the PyTorch library.
- **Job Submission**: 100 jobs were submitted, each training for 5 epochs. Upon completion, the environment was cleaned to free up storage resources.

