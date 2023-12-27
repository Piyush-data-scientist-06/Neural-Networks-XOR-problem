# Neural-Networks-XOR-problem
## Objective: 
To deepen the understanding of neural networks (forward & backward propagation, activation & loss functions) by implementing one from scratch in Python and gain practical hands-on experiential learning.
# Tasks Performed:
**1. Code Analysis:**
Examine the ann.py and layers.py files from the GitHub repository. Understand how different layers are implemented and how the network is trained.

**2. Implementation Challenge:**
Using the concepts and code samples as a guide, implement a neural network in Python. The network should include:
- At least one hidden layer.
- An activation function of your choice (e.g., ReLU, Sigmoid, Tanh).
- A loss function (e.g., Mean Squared Error, Binary Cross-Entropy). <br>
Document the  code thoroughly to explain your implementation choices. <br>

**3. Experimentation:**
- Train your neural network on a simple dataset (e.g., XOR problem, MNIST digits). <br>
- Experiment with different configurations (e.g., number of layers, types of activation functions, learning rates) and observe how they affect the performance of your network.
- Write a report summarizing the findings and reflect on the challenges faced during the execution of the tasks.

# Implementation Challenge:
**Implementing a Modular Neural Network for the XOR Problem:**

The construction of a neural network specifically designed to solve the XOR problem. The XOR (exclusive or) problem is a classic problem in neural network literature, which involves determining the output of an XOR gate given two binary inputs. This problem is especially significant as it represents a case where the relationship between the input and output cannot be captured by a simple linear model, thus requiring a neural network with at least one hidden layer to solve.

**Key Components of the Implemented Code:**

**Classes Defined:**

1. NeuralLayer: A base class for different types of neural network layers, enforcing the implementation of forward and backward methods in derived classes.
2. DenseLayer: This represents a fully connected neural layer (akin to FCLayer in some texts) that connects each input node to every output node, typical of traditional neural networks. 
3. ActivationLayer: Applies an activation function to the output of a neural layer. This implementation uses the Tanh function as the activation function, introducing non-linearity essential for complex patterns like the XOR relationship. 
4. NeuralNetwork: Represents the neural network itself, coordinating the flow of data through various layers and managing the training process.

**Functions Defined:**

1. activation_tanh and derivative_tanh: The hyperbolic tangent function and its derivative, introducing non-linearity essential for learning the XOR function.
2. loss_mse and loss_mse_derivative: The Mean Squared Error function and its derivative, a common choice for regression and binary classification tasks, measuring the average squared difference between the estimated values and the actual value. 
3. train (model_train): A method to train the neural network using the provided dataset. It iteratively adjusts the network's weights through forward and backward propagation.
4. predict (model_predict): A method for generating predictions using the trained network. It's crucial for evaluating the network's performance on the XOR problem.

# Experimentation Process and Results:
**Original Setup:** The network with 3 neurons in the hidden layer, trained with a learning rate of 0.1, successfully learned the XOR function, achieving low loss and producing accurate predictions after 1000 epochs.

**Experiment 1 (Varying Hidden Layer Neurons):** The hidden layer was tested with 2 and then 5 neurons. Both configurations resulted in the network failing to learn the XOR function correctly, as indicated by the similar outputs for all input combinations.

**Experiment 2 (Varying Learning Rates):** Learning rates of 0.01, 0.1, and 1.0 were tested. A learning rate of 0.01 resulted in slower learning, and 1.0 caused the network to fail, with extreme output values. The moderate rate of 0.1 was found to be optimal in this context.

**Insights and Learnings:** The experiments underscored the importance of choosing the right architecture and hyperparameters for neural network performance. Specifically, they highlighted the sensitivity of the network to the number of neurons in the hidden layer and the learning rate. While a learning rate that's too low slows down learning significantly, a rate that's too high can prevent the network from learning at all. The optimal number of neurons and learning rate successfully captured the XOR function, demonstrating the ability of a simple neural network to solve non-linear problems with proper configuration.
