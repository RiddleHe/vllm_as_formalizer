# Evaluations

## Check plan success

One of the most important validation of a solved plan is that it gets to the specified goal from the specified initial states in the `.pddl` problem file. 

To enable this, call the following script to check for final success and intermediate states as each action step executes:

```
python find_goal.py run [domain.pddl] [problem.pddl] [plan.txt]
```

If you want to quickly validate the script work, you can call `python find_goal.py test` for a blocksworld example.