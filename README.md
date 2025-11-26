# Genetic Algorithm for CNN Hyperparameter Optimization

A minimal example that uses a genetic algorithm to search for simple convolutional neural network (CNN) hyperparameters on MNIST. The script (main.py) generates a population of candidate CNN architectures and iteratively evolves them using selection, crossover and mutation, then trains and evaluates the best models.

Key points:
- Uses TensorFlow / Keras to build and train models.
- Evaluates candidate architectures using a short fitness training (2 epochs).
- Retrains the best candidate for more epochs (10 epochs) and records results.

### Requirements
- Python 3.8+
- tensorflow
- numpy

### Installation

> Create a virtual environment and install dependencies:

```sh
pip install tensorflow numpy
```

### Usage
Run the script: `python main.py`

### Configurable parameters
- KERNEL_FILTERS: list of filter sizes to choose from (e.g., [16, 32, 64, 128])
- KERNEL_SIZE: list of kernel widths (e.g., [3, 5])
- DENSE_UNITS: number of units for the final dense layer (e.g., [32, 64, 128, 256])
- QUANTITY_OF_LAYERS: number of Conv2D + MaxPool blocks per model
- POPULATION_SIZE: number of individuals per generation
- MUTATION_RATE: probability for mutations
- GENERATIONS: number of evolution cycles


### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.