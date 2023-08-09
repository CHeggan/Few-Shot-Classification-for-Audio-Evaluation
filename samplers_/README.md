# Few-Shot Classification for Acoustic Signals Codebase
A brand new evaluation suite for a variety of few-shot acoustic classification tasks. 

This is an active codebase in that we will be using and updating it alongside our primary research work behind he scenes. Each release of a new work will likely see an update or new branch of this evaluation suite. 



## News & Updates
 - 9/8/23: Public release of the codebase 


## Citation
This codebase was developed for our most recent works [MT-SLVR](https://github.com/CHeggan/MT-SLVR) and [MetaAudio](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark). If you find this repo useful or utilise it in your work, please consider citing our works:

```
@misc{heggan2023mtslvr,
      title={MT-SLVR: Multi-Task Self-Supervised Learning for Transformation In(Variant) Representations}, 
      author={Calum Heggan and Tim Hospedales and Sam Budgett and Mehrdad Yaghoobi},
      year={2023},
      eprint={2305.17191},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@InProceedings{10.1007/978-3-031-15919-0_19,
author="Heggan, Calum
and Budgett, Sam
and Hospedales, Timothy
and Yaghoobi, Mehrdad",
title="MetaAudio: A Few-Shot Audio Classification Benchmark",
booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2022",
year="2022",
publisher="Springer International Publishing",
pages="219--230",
isbn="978-3-031-15919-0"
}

```


# Few-Shot Evaluation
In this framework we evaluate over few-shot classification tasks, each containing a support set and a query set. These are best though about as mini learning problems in the regime where you have limited labelled data. The support set is effectively a mini-training set, which we use to train a classifier, and the query set acts as analogously to a tests set. 

In simplified code: For each pre-trained model and each considered dataset, the evaluation framework does the following:
```python 
# Consider some number n of few-shot classification tasks, we use n = 10,000
for _ in range(num_few_shot_tasks):
    # Sample new fs task function grabs a new support and query set making a new task
    x_support, y_support, x_query, y_query = sample_new_fs_task()
    # Generate feature embeddings for both our supports and queries using our frozen model
    sup_feats = model(x_support)
    quer_feats = model(x_queries)

    # Train our classifier on the support features
    classifier.train(sup_feats, y_support)

    # Test our trained classifier on our queries
    task_accuracy = classifier.test(quer_feats, y_query)

# We then average over all of the individual task accuracies for this dataset and model combination in order to obtain a mean and std/CI
```



# Using the Repo
## Contents & Variables


## Environment


## Datasets & Processing


## Example Run



# Functionality
## Variable Length Handling

## Input Data Dimensionality


## Main Model Loop


## Conversion to Features 


## Types of Tasks
hard/easy etc


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