# Pipeline
Frame it as a blend between the current eval codebase and the vv codebase. We only perform evaluation, there is no training or fine-tuning here.

Have a trained models folder that we loop over (similar as is found in FS eval codebase)

Generate a compiled set of features and labels (like in vv)

Use mostly same code from vv after that


Still to do:
 - variable length sklin classifier
 - obtain metrics for accuracies (std etc)
 - bundle difficulty results
 - proper speed testing
 - output to file