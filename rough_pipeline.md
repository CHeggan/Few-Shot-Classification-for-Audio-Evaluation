# Pipeline
Frame it as a blend between the current eval codebase and the vv codebase. We only perform evaluation, there is no training or fine-tuning here.

Have a trained models folder that we loop over (similar as is found in FS eval codebase)

Generate a compiled set of features and labels (like in vv)

Use mostly same code from vv after that

```
For model in model:
    for dataset in datasets:
        create dataset with only relevant subsplit of data (minimise overhead)
        generate data features
        for task_hardness in [easy, avg, hard]:
            perform classification on n tasks
```

- needs to deal with variable input into feature extraction


## Args
Args needed:
    - dataset(s)
    - n
    - k
    - test tasks
    - classifier
    - task hardness