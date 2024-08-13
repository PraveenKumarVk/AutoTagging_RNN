# Auto Tagging using RNN

This project focuses on implementing an automatic tagging system using Recurrent Neural Networks (RNNs). The aim is to automatically generate tags for text data, which can be used in various applications such as content classification, recommendation systems, and information retrieval.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to build a model that can automatically generate relevant tags for a given text input. The model is built using a Recurrent Neural Network (RNN) with Keras, a deep learning library for Python. The tagging system is designed to improve the accessibility and searchability of textual content by assigning meaningful tags.

## Dataset

The dataset used for training the model includes textual data along with associated tags. Each text entry in the dataset is paired with a set of tags that describe the content. The dataset is split into training, validation, and test sets to evaluate the model's performance.

## Project Structure

```plaintext
Auto_Tagging_RNN/
├── AutoTagging.ipynb         # Jupyter Notebook containing the code
├── AutoTagging_data.zip      # Compressed Directory containing the dataset
├── weights.best.keras        # Saved trained models
└── README.md                 # This README file
```

## Usage

To use the model for auto-tagging, follow these steps:

1. **Prepare the Data:** Ensure that your dataset is in the correct format and is placed in the `data/` directory.
2. **Train the Model:** Open the `Auto_Tagging_RNN.ipynb` notebook and run the cells to train the model on your dataset.
3. **Evaluate the Model:** After training, evaluate the model's performance using the test data.
4. **Generate Tags:** Use the trained model to generate tags for new text inputs.

## Model Architecture

The model is built using a Recurrent Neural Network (RNN) with the following architecture:

* Embedding Layer: Converts text into dense vectors of fixed size.
* LSTM Layer: Captures the sequential information in the text.
* Dense Layer: Outputs the probability of each tag.

## Training

The model is trained using the following configuration:

* Loss Function: Binary Crossentropy
* Optimizer: Adam
* Batch Size: 32
* Epochs: 10

Hyperparameters such as the learning rate, batch size, and number of epochs can be adjusted in the notebook.

## Evaluation

The model is evaluated using standard metrics such as precision, recall, and F1-score. The evaluation results are saved in the `results/` directory.

## Results

The model achieves the following performance metrics on the test set:

* Precision: 0.80
* Recall: 0.79
* F1-Score: 0.80

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. Any contributions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
