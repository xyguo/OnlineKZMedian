# Local search algorithm for online *k*-median

## Structure

* `OnlineKZMed`: The main algorithm implemented as a class in `okzm/okzm.py`.
* `Assignmenter`: The data structure that maintains a clustering solution and supports (not yet) fast nearest-neighbor query.

## Progress

1. [X] Unit tests for all basic functions
2. [ ] Use heap in the implementation of class `Assignmenter` to support faster query/modification of nearest facility information 
3. [ ] Implement the `Assignmenter` class via Cython to accelerate.  

The profiling result shows that the major time-consuming part is in `Assignmenter.can_swap` (occupies > 95% total time), which consists of mainly the following parts:
* `DistanceMatrix.pairwise_dist_argmin_min`: 53%. 
* `Assignmenter.nearest_facility`: 15%
* Rest: 27%

## Requirements

* `numpy` >= 1.16.4
* 0.23 >= `scikit-learn` >= 0.21.2

## How to run it

The main experiment code is in `experiment_neurips20.py`, to run it, just execute the command:

```shell
python experiment_neurips20.py
```

To suppress to verbose output, set variable value of `verbose=False` in the experiment file.
