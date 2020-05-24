# Local search algorithm for online *k*-median

## Structure

* The main algorithm is implemented as a class `OnlineKZMed` in `okzm/okzm.py`.
* The data structure that supports 

## Progress

1. [X] Unit tests for all basic functions
1. [ ] Use heap in the implementation of class `Assignmenter` to support faster query/modification of nearest facility information 
2. [ ] Implement the `Assignmenter` class via Cython to accelerate.  

## Requirements

* `numpy` >= 1.16.4
* 0.23 >= `scikit-learn` >= 0.21.2

## How to run it

The main experiment code is in `experiment_neurips20.py`, to run it, just execute the command:

```shell
python experiment_neurips20.py
```

To suppress to verbose output, set variable value of `verbose=False` in the experiment file.
