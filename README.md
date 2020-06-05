# Local search algorithm for online *k*-median with outliers

## Structure

* `OnlineKZMed`: The main algorithm implemented as a class in `okzm/okzm.py`.
* `Assignmenter`: The data structure that maintains a clustering solution and supports (not yet) fast nearest-neighbor query. See `okzm/assignmenter` for details.

## Progress

1. [X] Unit tests for all basic functions
2. [X] ~~Use heap in the implementation of class `Assignmenter` to support faster query/modification of nearest facility information~~ Discarded: this is not a bottleneck. 
3. [X] ~~Implement the `Assignmenter` class via Cython to accelerate.~~ Discarded: pure Python is already fast enough.

The profiling result shows that the major time-consuming part is in `Assignmenter.can_swap` (occupies > 95% total time).

## Requirements

* `numpy` >= 1.16.4
* 0.23 >= `scikit-learn` >= 0.21.2

## To reproduce our experiment result

**WARNING**: we don't include the data set file in this code repository, so you need first download the data by yourself and put it in the correct path. See `utils/get_data.py` for details. But you can still try the code on synthesized data. 

---

The main experiment code is in `experiment_for_neurips20.py`, to run it, just execute the command:

```shell
python experiment_for_neurips20.py
```
This code records the recourse and cost of our algorithm. To suppress to verbose output, set variable value of `verbose=False` in the experiment file.

To obtain the graph for approximation ratio, one need also estimate the offline OPT. This can be done by running
```shell
python experiment_estimate_opt.py
```

