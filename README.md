# DeaSTL

This repository contains the implementation code, trained model for our work: An EEG-based Dual-Stream Spatial-Spectral-Temporal Large Model for Self-Limited Epilepsy with Centrotemporal Spikes. 

## Introduction

Self-limited epilepsy with centrotemporal spikes (SeLECTS) is the most common form of focal epilepsy in childhood, accounting for 20% to 25% of all childhood epilepsy cases and may be associated with cognitive dysfunction and behavioral issues. Accurate detection and assessment of epileptic discharges in EEG signals, particularly the spike-wave index (SWI), are crucial for timely intervention and treatment. Manual analysis of EEG data is labor-intensive and prone to errors, underscoring the need for automated methods. In the present study, we propose a novel Dual-Stream Spatial-Spectral-Temporal Large model (DeaSTL) that leverages a large-scale EEG architecture to effectively capture the multidimensional characteristics of EEG signals associated with SeLECTS syndrome. Our model integrates multi-view temporal representations and spatial-spectral representations through a dual-stream approach, enhancing the learning of complex patterns in EEG data. We introduce the SJTU SeLECTS EEG Dataset (SLED), a comprehensive EEG dataset from 212 patients diagnosed with SeLECTS, including annotations for abnormal discharge detection, wake-sleep period classification, and SWI estimation. Addressing the previously unexplored problem of SWI prediction, we provide a novel method for quantifying the severity of epileptic discharges during sleep. Extensive experiments demonstrate that our DeaSTL model significantly outperforms several state-of-the-art methods across multiple tasks, showcasing its potential for clinical application in assisting diagnosis and treatment planning.

## Requirements

python==3.11

pytorch==2.0.1 

torchvision==0.15.2 

torchaudio==2.0.2 

pytorch-cuda=11.8


Install dependencies:

```
pip install -r requirements.txt
```


## Test

`modeling_finetuning.py` contains the DeaSTL model we proposed.

To protect patient privacy and comply with hospital regulations, we recommend using your own data and adding the appropriate path to your EEG dataset for testing. To test DeaSTL, run

```
run.sh
```


## Pre-trained Models

`checkpoints/cts/checkpoints.pt`, `checkpoints/sleep/checkpoints.pt`, `checkpoints/swi/checkpoints.pt` are the trained model of DeaSTL for three tasks.
