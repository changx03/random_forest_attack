# Algorithm for Random Forest Attack

1. Initialize maximum budget (the max. perturbation is allowed.), the decision trees, and the input sample which we want to mutate.
1. Building nodes by traversing each decision tree (one tree one path)
1. Finding the node with minimum cost to switch the decision (Using the last node. *Note: Can be any node in the graph. Top nodes are also viable choices.*)
1. Updating budget and input. Checking the new prediction. If the prediction is switched to the desired class, returns the updated input and stop. Else, builds nodes with updated input.

## Problem

Exist the current branch when there is not enough budget to spend.

## Pseudocode

* Input: 
    * X: Input example
    * y: Output label 
    * estimators: Multiple Decision Trees in a RF
    * budget: Maximum perturbation
* Output: 
    * X_adv: Adversarial example

```python
x_stack = [X]
directions = [zeros_of_number_of_features] # Keeps the direction of updates
while predict(X) == y and budget > 0:
    paths = []
    for estimator in estimators:
        path = build_path(x_stack[-1], estimator)
        paths.append(path)
    
    # Find optimal update
    x_next, path, direction = find_optimal_update(paths, directions[-1])
    if predict(x_next) != y:
        return x_next # Found x_adv. Done.
    x_stack.append(x_next)
    directions.append(direction)
    budget -= path.cost

    
```
