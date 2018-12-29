# TV Script Generator

## Project Overview

In this project, I generated my own [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using RNNs. I used part of the [Seinfled dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons. The Neural Network I generated a new, "fake" TV script, based on patterns it reconginzes in this training data.

## Project Instruction

### Instruction

1. Clone the repository and navigate to the downloaded folder.
	```
		git clone https://github.com/PradeepVenna292/Udacity-Deep-Learning-Nanodegree-Project3-Generating-TV-Scripts.git
		cd TV-Script-Generator
	```
2. Open the `dlnd_tv_script_generation.ipynb` file. Of course, you can find HTML version of the file.
	```
		jupyter notebook dlnd_tv_script_generation.ipynb
	```
3. Read and follow the instructions! This repository already includes the dataset in a form of txt flie in `data` folder.

## Project Information

### Contents

- Get the Data
- Explore the Data
- Implement Pre-processing Functions
	- Lookup Table
	- Tokenize Punctuation
- Pre-process all the data and save it
- Check Access to GPU
- Input
	- Batching
	- Test your dataloader
	- Sizes
	- Values
- Build the Neural Network
	- Define forward and backpropagation
- Neural Network Training
	- Train Loop
	- Hyperparameters
	- Train 
- Generate TV Script
	- Generate text
	- Generate a new script

### Model
| Layer | Input Dimension | Output Dimension |
| ----- | --------------- | ---------- |
|Embedding|Vocab Size| 200 |
|LSTM|200|256|
|FC|256|Vocab Size|

### Hyperparameters
|Data Parameter|Value|
|--------------|------|
|sequence_length| 10|
|batch_size| 128 |

|Training Parameter|Value|
|------------------|-----|
|num_epochs|20|
|learning_rate|0.001|
|embedding_dim|200|
|hidden_dim|256|
|n_layers(Number of RNN Layers)|2|

### Libraries

The list below represents main libraries and its objects for the project.
- [PyTorch](https://pytorch.org) (LSTM)

