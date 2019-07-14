# Time2Vec: Learning a Vector representation of Time

This is an attempt of reproducing the paper !["Time2Vec: Learning a Vector Representation of Time"](https://arxiv.org/pdf/1907.05321.pdf) in PyTorch.

## Summary
Popular activation functions are unable to capture periodicity of the input, hence they cannot capture the periodic nature of Time and Dates.

Gradients of Activations like ReLU, Softmax, Sigmoid etc. either explode or plateu due to the input continuosly growing.

Currently, a synthesized dataset of integers is used to test the functionality of the proposed method. 

I plan to add more experiments to this repository.

## Prerequisites

PyTorch (Tested on PyTorch 1.1 on Python 3.6)

## Steps for running experiments:
    1. Clone this repository, and change the directory to the folder.
    2. To start training, In a command line, enter: "python3 experiment.py"
    3. To use the Periodic Activation layers in your projects, copy the file "periodic_activations.py" and import: 
        "from periodic_activations import SineActivation"

## To-Do
    1. Adding experiments on popular datasets
    2. Adding couple of Periodic activations like Triangle function and Modulo
    3. Adding comparisons to Aperiodic activations.

## Authors

***Surya Kant Sahu** - [ojus1](https://github.com/ojus1)

## License

This project is licensed under the MIT License - [LICENSE.md](./LICENSE.md)

## Acknowledgments

* Packages used for Machine learning model: Python: Pandas, PyTorch
* Paper Title: "Time2Vec: Learning a Vector Representation of Time" - https://arxiv.org/pdf/1907.05321.pdf