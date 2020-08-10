# Algorithm for Random Forest Attack

1. Initialize maximum budget (the max. perturbation is allowed.), the decision trees, and the input sample which we want to mutate.
1. Building nodes by traversing each decision tree (one tree one path)
1. Finding the node with minimum cost to switch the decision (Using the last node. _Note: Can be any node in the graph. Top nodes are also viable choices._)
1. Updating budget and input. Checking the new prediction. If the prediction is switched to the desired class, returns the updated input and stop. Else, builds nodes with updated input.

## Problem

Exist the current branch when there is not enough budget to spend.

## Pseudo code

- Inputs:
  - k: number of Decision Trees in a RF
  - X: (1\*m Array) Input example
  - y: (int) Output label
  - estimators: (Array) An array with k Decision Trees
  - budget: (float) Maximum perturbation
- Outputs:
  - X_adv: (1\*m Array) Adversarial example
- Parameters:
  - j: (int) number of updates
  - paths: (Array) An array with k Path instances
  - update_values: (Array) An array with j elements. It keeps the update values (Can be positive and negative)
  - update_feature_indices: (Array) An array with j elements. It keeps the feature index

```python
update_feature_indices = [0] # Root index doesn't matter, because the update value is 0.
update_values = [0]

paths = []
for estimator in estimators:
    path = build_path(x_stack[-1], estimator)
    paths.append(path)

# Find optimal update
path_index, feature_index, update_value = find_optimal_update(
    paths,
    update_feature_indices,
    update_values)

if (budget - abs(update_value)) <= 0: # Minimum change out of budget, switch node
    paths[path_index].cost = infinity # No longer use this path
    continue
budget -= abs(update_value)
update_feature_indices.append(feature_index)
update_values.append(update_value)
x_next = update_X(X, update_feature_indices, update_values)
if predict(x_next) != y:
    return x_next # Found x_adv. Done.
```
