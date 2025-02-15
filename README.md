# Reuters Newswire Classification Project

## Overview
This project uses a Recurrent Neural Network (RNN) with Bidirectional LSTM layers to classify newswire articles from the Reuters dataset into 46 different categories. The model is built using TensorFlow and Keras, and it aims to provide accurate text classification for the given newswire articles.

## Features
- **Embedding Layer**: Converts words into dense vector representations.
- **Bidirectional LSTM Layers**: Capture information from both forward and backward sequences.
- **Dropout Layers**: Prevent overfitting by randomly dropping units during training.
- **Dense Layers**: Used for classification with ReLU activation.
- **Output Layer**: Uses softmax activation for multi-class classification.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/davoodoli/LSTM_RNN_News_Dataset.git
