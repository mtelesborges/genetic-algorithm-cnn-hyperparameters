import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random

def create_individual(quantity_of_layers: int,
                      kernel_filters: list[int],
                      kernel_size: list[int],
                      dense_units: list[int]):
  _layers = []

  for _ in range(quantity_of_layers):
    _layers.append({
        "kernel_filters": random.choice(kernel_filters),
        "kernel_size": random.choice(kernel_size),
    })

  return {
      "layers": _layers,
      "dense_units": random.choice(dense_units)
  }

def create_population(population_size: int,
                      quantity_of_layers: int,
                      kernel_filters: list[int],
                      kernel_size: list[int],
                      dense_units: list[int]):
  population = []

  for _ in range(population_size):
    individual = create_individual(quantity_of_layers, kernel_filters, kernel_size, dense_units)
    population.append(individual)

  return population

def build_model(individual):

  configuration = []

  for j, value in enumerate(individual['layers']):
    if j == 0:
      configuration.append(
          layers.Conv2D(value['kernel_filters'],
                        (value['kernel_size'], value['kernel_size']),
                        activation='relu',
                        input_shape=(28, 28, 1))
      )
    else:
      configuration.append(
          layers.Conv2D(value['kernel_filters'],
                        (value['kernel_size'], value['kernel_size']))
      )
    configuration.append(layers.MaxPooling2D((2, 2)))

  configuration.append(layers.Flatten())
  configuration.append(layers.Dense(individual['dense_units'], activation='relu'))
  configuration.append(layers.Dense(10, activation='softmax'))

  model = models.Sequential(configuration)

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model

def fitness(population, train_images, train_labels):
  scores = []

  for i in range(len(population)):
    individual = population[i]

    model = build_model(individual)

    history = model.fit(
      train_images, train_labels,
      epochs=2,
      validation_split=0.2,
      verbose=0
    )

    accuracy = history.history["val_accuracy"][-1]

    scores.append((accuracy, individual))
    print(f"Individual: {individual}  => accuracy={accuracy:.4f}")

  return scores

def evolve_population(scores,
                      population_size: int,
                      quantity_of_layers: int,
                      kernel_filters: list[int],
                      kernel_size: list[int],
                      dense_units: list[int],
                      mutation_rate: float):
  scores.sort(reverse=True, key=lambda x: x[0])
  elites = [individual for _, individual in scores[:2]]

  new_population = elites.copy()

  while len(new_population) < population_size:
      p1, p2 = random.sample(elites, 2)
      child = crossover(p1, p2)
      child = mutate(child, quantity_of_layers, kernel_filters, kernel_size, dense_units, mutation_rate)
      new_population.append(child)

  return new_population

def crossover(parent1, parent2):
  child = {'dense_units': 0, 'layers': []}

  child['dense_units'] = parent1['dense_units'] if random.random() < 0.5 else parent2['dense_units']

  for i, value in enumerate(parent1['layers']):
      l = parent1['layers'][i] if random.random() < 0.5 else parent2['layers'][i]
      child['layers'].append(l)

  return child

def mutate(individual,
           quantity_of_layers: int,
           kernel_filters: list[int],
           kernel_size: list[int],
           dense_units: list[int],
           mutation_rate: float):

  if random.random() < mutation_rate:
    individual['dense_units'] = random.choice(dense_units)

  for i in range(quantity_of_layers):
    if random.random() < mutation_rate:
      individual['layers'][i]['kernel_filters'] = random.choice(kernel_filters)

    if random.random() < mutation_rate:
      individual['layers'][i]['kernel_size'] = random.choice(kernel_size)

  return individual

KERNEL_FILTERS = [16, 32, 64, 128]
KERNEL_SIZE = [3, 5]
DENSE_UNITS = [32, 64, 128, 256]
QUANTITY_OF_LAYERS = 2
POPULATION_SIZE = 5
MUTATION_RATE = 0.2
GENERATIONS = 5

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

population = create_population(POPULATION_SIZE, QUANTITY_OF_LAYERS, KERNEL_FILTERS, KERNEL_SIZE, DENSE_UNITS)

results = []

for i in range(POPULATION_SIZE):
  print(f"\n=== Generation {i+1} ===")

  scores = fitness(population, train_images, train_labels)

  parameters = scores[0][1]

  model = build_model(parameters)
  model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

  loss, accuracy = model.evaluate(test_images, test_labels)
  results.append({
      "accuracy": accuracy,
      "loss": loss,
      "parameters": parameters
  })

  population = evolve_population(scores, POPULATION_SIZE, QUANTITY_OF_LAYERS, KERNEL_FILTERS, KERNEL_SIZE, DENSE_UNITS, MUTATION_RATE)