# Variable Tracking for Python

## Description
The variable tracking extension for python includes functionality for backtracking if invalid variables are used. In such a case the parser will backtrack automatically and continue generating until it can no longer detect any usage of undefinded variables

## Usage
In order to use the variable tracking extension the Itergen object needs to be created with the ```python_var_tracking grammar```. The parser is automatically selected if the Itergen object is created with the changed grammar.
The parser enforces variable consistency by keeping track of variables that have been defined up until a certain point and then checks each usage for a prior definition. In oder to allow for prior definition of custom existing variables,
we have provided two new parameters to the Itergen class that allow for costumizationÖ
- ``predefined_variables``: This parameter allows the user to set a list of strings that is automatically whitelisted for use as variables to provide compartibility with preexisting code.
This list always includes python built-in functions.
- ``backtracking_allowed``: This parameter toggles the automatic backtracking in case invalid output should be displayed for debugging purposes
- ``backtracking_leniancy_tokens``: This parameter controls the amount of tokens that are allowed to be generated before backtracking is triggered by a detected invalid usage. This delay prevents unnecessary backtracking due to some variables being incorrectly classified while the statement is still being generated. For further information on this topic please refer to the syncode paper and the remainder technology used.

## Example
For an example on how all this looks in practice, please have a look at the Testscenario notebook in the project's root folder

