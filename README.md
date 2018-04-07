# SemEval-2018-task12
This repository contains winner system (GIST team) code for SemEval 2018 task 12: Argument Reasoning Comprehension

The paper for task description can be checked at [The Argument Reasoning Comprehension Task: Identification and Reconstruction of Implicit Warrants](https://arxiv.org/pdf/1708.01425.pdf) by Habernal et al. (2018).

The official result (of 22 participants) can be checked at [here](https://github.com/habernal/semeval2018-task12-results)

Our paper is GIST at SemEval-2018 Task 12: A network transferring inference knowledge to Argument Reasoning Comprehension task

## Code description
We changed original raw data into preprocessed data in advance, so this code use the preprocessed data.
So this repo dosen't contain some code about preprocessing.
And, All hyper parameter is set to be default value, which is same with our paper.
You can check our code by just typing 'python GIST_hongking9.py' in command line.

## Library requirement
theano, lasagne, pickle, numpy, os, time, mkl

## Our Enviroment
Window 10, python 2.7

