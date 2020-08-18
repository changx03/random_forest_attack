# Algorithm for Random Forest Attack

## Parameters

- Inputs:

  - k: Number of Decision Trees in a RF
  - m: Number of input features
  - x: A single data point
  - y: The output label for the given input
  - model: A trained Random Forest classifier which contains k Decision Trees
  - budget: Maximum perturbation (Default=0.1\*m)

- Outputs:

  - X_adv: (1\*m Array) Adversarial example

- Parameters:

  - x_stack: LIFO stack. Keeps tracking the updates of x
  - directions: The directions of features of x that is allowed to move (-1: Negative only, 0: Both, 1: Positive only)
  - paths_stack: LIFO stack. Keeps tracking the selected paths with given x

## Pseudocode

```python
initialize x_stack, push x
initialize directions, default to zeros
find initial paths
initialize paths_stack, push initial paths

while True:
    if prediction(x_stack.peek) is not y:
        return x_stack.peek

    find a path in paths_stack.peek
    if path is None and length(paths_stack) == 1:
        break

    while path is None or budget < 0:
        if length(paths_stack) > 1:
            last_x = x_stack.pop
            paths_stack.pop
        else:
            last_x = x

        restore directions
        refund budget
        find a path in paths_stack.peek

        if path is None:
            return x_stack.peek

    compute next_x based on the selected path
    x_stack.push(next_x)
    reduce budget
    update directions
    set the node in the selected path as visited
    compute new paths based on next_x
    paths_stack.push(paths)

return x_stack.peek
```
