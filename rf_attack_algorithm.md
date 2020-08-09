# Algorithm for Random Forest Attack

1. Initialize maximum budget (the max. perturbation is allowed.), the decision trees, and the input sample which we want to mutate.
1. Building nodes by traversing each decision tree (one tree one path)
1. Finding the node with minimum cost to switch the decision (Using the last node. *Note: Can be any node in the graph. Top nodes are also viable choices.*)
1. Updating budget and input. Checking the new prediction. If the prediction is switched to the desired class, returns the updated input and stop. Else, builds nodes with updated input.

## Problem

Exist the current branch when there is not enough budget to spend.
