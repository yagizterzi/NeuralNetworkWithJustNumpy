# Neural Network from Scratch using NumPy

This project implements a simple neural network from scratch using only NumPy. The network is trained on the MNIST dataset to classify handwritten digits (0-9) and achieves an accuracy of **92% (0.92)** on the test set.

## Features
- Fully connected neural network built without any deep learning frameworks
- Uses only NumPy for all computations
- Trains on the MNIST dataset
- Implements forward and backward propagation
- Uses gradient descent with mini batches and cross entropy loss for classification

## Dataset
This dataset uses 10.000 of 60.000 images in MNSIT dataset which has handwritten digits, each 28x28 pixels in size. The dataset is publicly available and can be loaded using various libraries.
## Requirements
To run this notebook, install the following dependencies:

```bash
pip install numpy matplotlib
```

## Usage
1. Clone this repository:
   ```bash
   git clone <[repository_url](https://github.com/yagizterzi/NeuralNetworkWithJustNumpy)>
   cd <NeuralNetworkWithJustNump>
   ```
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Neural_Network.ipynb
   ```
3. Follow the notebook steps to train and evaluate the model.

## Model Architecture
- **Input Layer:** 784 neurons (flattened 28x28 images)
- **Hidden Layer:** 128 neurons (ReLU activation)
- **Output Layer:** 10 neurons (Softmax activation)

## Training Details
- **Optimizer:** Gradient Descent
- **Loss Function:** Cross-Entropy
- **Epochs:** Configurable
- **Batch Size:** Configurable
- **Final Accuracy:** **92% (0.92)** on the MNIST test set

## Results
After training, the model achieves an impressive accuracy of **92%** on the MNIST test set, demonstrating the effectiveness of a simple neural network implemented with NumPy.

## Contributions
Feel free to open issues and submit pull requests to improve this implementation.

## License
This project is open-source and available under the MIT License.



