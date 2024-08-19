# Few-Shot Classification for Acoustic Signals Codebase
A brand new evaluation suite for a variety of few-shot acoustic classification tasks. 





## News & Updates
 - 9/8/23: Public release of the codebase 
 - 19/8/24: Codebase update to include work on transfer learning


## Upcoming Updates
This codebase will be updated with code for two of our recent works:
 - "On the Transferability of Large-Scale Self-Supervision to Few-Shot Audio Classification" ICASSP SASB 2024
    - Compatibility with some pre-trained large-scale self-supervised speech models
    - Variety of pass through options as per the original models
 - "From Pixels to Waveforms: Evaluating Pre-trained Image Models for Few-Shot Audio Classification" IJCNN 2024
    - Inclusion of pre-train image models
    - These include both supervised pre-train (mainly fromm torchvision) and self-supervised (obtained from a variety of sources)
    - Various audio input and normalisation options so that they can be used with audio models

## Other TODO:
There is still some work to be done for this codebase:
 - Notes on how to setup and format included datasets
 - Inclusion of a guide on how to setup custom datasets
 - Reproducible environment Release



## Citation 
This codebase was initially developed for our most recent works [MT-SLVR](https://github.com/CHeggan/MT-SLVR) and [MetaAudio](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark). If you find this repo useful or utilise it in your work, please consider citing our works:

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

We also include code relating to our works [Pixels to Waveforms](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=_1R5aa8AAAAJ&citation_for_view=_1R5aa8AAAAJ:UeHWp8X0CEIC) and [On the Transferability of Large-Scale Self-Supervision...](https://arxiv.org/abs/2402.01274). If you use the transfer learning apsect of this codebase, please consider citing teh following:

```
@inproceedings{f71e00d39bfe40739c3aeddbff1a1037,
title = "From pixels to waveforms: Evaluating pre-trained image models for few-shot audio classification",
author = "Calum Heggan and Hospedales, {Timothy M.} and Sam Budgett and {Yaghoobi Vaighan}, Mehrdad",
year = "2024",
month = mar,
day = "15",
language = "English",
booktitle = "Proceedings of the International Joint Conference on Neural Networks (IJCNN)",
note = "International Joint Conference on Neural Networks, IJCNN 2024 ; Conference date: 30-06-2024 Through 05-07-2024",
url = "https://2024.ieeewcci.org/",
}
```

```
@INPROCEEDINGS{10626094,
  author={Heggan, Calum and Budgett, Sam and Hospedales, Tim and Yaghoobi, Mehrdad},
  booktitle={2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)}, 
  title={On the Transferability of Large-Scale Self-Supervision to Few-Shot Audio Classification}, 
  year={2024},
  volume={},
  number={},
  pages={515-519},
  keywords={Correlation;Conferences;Self-supervised learning;Benchmark testing;Signal processing;Feature extraction;Acoustics;Self-Supervision;Few-Shot Learning},
  doi={10.1109/ICASSPW62465.2024.10626094}}
```


# Few-Shot Evaluation
In this framework we evaluate over few-shot classification tasks, each containing a support set and a query set. These are best though about as mini learning problems in the regime where you have limited labelled data. The support set is effectively a mini-training set, which we use to train a classifier, and the query set acts as analogously to a tests set. 

<img src="repo_images/Few_shot_audio_classification.svg">
Example of 3 5-way 1-shot audio tasks, with support sets (left) and query sets (right). 


In simplified code, for each pre-trained model and each considered dataset, the evaluation framework does the following:
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
The evaluation suite contains a variety of options for more advanced evaluation (details and more discussion of each point is given later in the [Functionality](#functionality) Section):
 - Many supported datasets
 - Multiple classifiers
 - Main codebase is written model agnostically
 - Few-Shot task variant options 


## Environment
We are currently working on re-factoring the environment for this codebase. With the inclusion of transfer learning components (in particular from torchvision and s3prl), a suitable fits all environment can be tricky, and so we recommend builing a separate enviroment for each main evaluation protocol for the codebase (1 for basic eval of models, 1 for image-transfer, 1 for speech and s3prl). 

## Datasets & Processing
We include support for 10 total datasets for evaluation. Many of these are from the [MetaAudio](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark/tree/main) work, however some were added in [MT-SLVR](https://github.com/CHeggan/MT-SLVR). Datasets currently supported by default are:
 - NSynth
 - ESC-50
 - BirdClef Pruned
 - VoxCeleb1
 - FSDKaggle18
 - Speech Accent Archive (Test Only)
 - Common Voice Delta Segment v12 (Test Only)
 - Crema-D (Test Only)
 - SpeechCommands v2 (Test Only)
 - Watkins Marine Mammal Database (Test Only)

These datasets are supported in that they have included train/val/test splits and statistics information included within the codebase. The actual datasets will have to be manually downloaded and setup prior to running this framework. We do not yet include a comprehensive guide here on how to do this but we refer you to our previous notes in the MetaAudio repo on how to do this [here](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark/tree/main/Dataset%20Processing). 

The set of datasets evaluated over will depend on the value of the 'split' argument passed during the script call. If split is set as 'val' only datasets that have a validation split can and will be used. If split is set to 'test', the test split of every possible dataset will be used. This separation allows for some hyper parameter tuning of models on a so-called valuation split, similarly to how it would be done in meta-learning.


## Additional Setup
There are a few blank folders to be added to the main repo folder:
 - "TRAINED": The folder which will contain the trained models we want to evaluate over
 - "RESULT FILES": Folder to contain the results of the few-shot eval run

Within "TRAINED", unless using a single model, models will have to be nested within additional folders. Our code utilises a model loop where any model within a given directory will be loaded and evaluated sequentially. This process is great for mass evaluation with similar model types (architecture/adapter/input etc) but can lead to errors if not handled properly.

## Example Run

```bash
python .\main.py  --model_name resnet18 --adapter None --num_splits 1 --model_fc_out 1000 --model_dir "X:/Trained_model_storage/MT-SLVR" --num_tasks 100 --split test --classifier "sklin" --results_file "nan_testing" --rep_length 1
```



# Functionality
## Variable Length Handling
Our evaluation datasets range from fixed length (where all samples are exactly the same size) to massively variable length (where samples range from 0.1 seconds to multiple minutes). 

To handle this we build the framework around the idea of variable representation length, where effectively given an input sample, we can choose how long it should be as input to the model. We implement this idea using the following constraints:
 - Any sample from any dataset can have its input length modified
 - If a sample is to be extended, it is circularly padded
 - If a sample is longer than what we want, we split it up into sections that are wanted length and randomly sample from them

All representation length changes are done before conversion to spectrograms (if applicable).

## Input Data Dimensionality
The codebase offers 3 different data input representations for models:
 - Raw 1d signal (dims=1, channels=1)
 - 2d 1-channel typical Mel-Spectrogram (dims=2, channels=1)
 - 2d 3-channel Mel-Spectrogram w/ channels 2 and 3 containing explicit phase and amplitude information (dims=2, channels=3)

## Main Model Loop
As the evaluation framework is largely built around the purpose of mass model evaluation, we utilise an inner loop for both available models and considered datasets. Tis comes with a few benefits and a few possible drawbacks. 

Benefits:
 - Allows many models to be evaluated over many datasets with a single script run

Drawbacks:
 - Models that are to be evaluated in a single script run must be of similar type, i.e same input type/base model architecture
 - If running single models, they have to be nested inside a folder

As of now, our code primarily supports resnet based architectures. If you want to use a base neural model that is different you will have to do the following:
 - Add the model construction scripts to the "models_" folder
 - Modify the encoder selection script to include your model
 - Modify the setup.py script to include loading your model based on some input name

## Conversion to Features 
Sampling many tasks from datasets and extracting/evaluating them sequentially can be very expensive, i.e 20 minutes for a single set of 10,000 tasks for a single model.

To alleviate this issue we implemented a feature extraction pipeline within the evaluation code. Effectively what this section does is, given an entire evaluation dataset that we want to sample few-shot tasks from, process the whole dataset into its feature representation. As we can now sample tasks from the already extracted feature space, the sampling and evaluation per task time is significantly faster. In a few specific cases, this approach of pre-extracting the full dataset is not the faster option, these are:
 - If only evaluating over a few tasks, i.e a few hundred
 - If the evaluation dataset is very large 

As there is no clear cut best solution, we implement a switch for each dataset within the config file which determines which approach should be taken, i.e. full dataset extraction or per-task sampling and extraction. 

## Types of Tasks
### Easy, Average and Hard Tasks
As noted by recent works ([here](https://arxiv.org/abs/2110.13953) and [here](https://openreview.net/forum?id=wq0luyH3m4)), evaluation over just 'average' tasks is fairly limited and does not give an entire picture into the effectiveness of a model. In the second of these works, the authors implement a greedy search approach to find 'easy' and 'hard' tasks from datasets. These tasks are effectively mined from the extracted features of a specific dataset for a specific model. 

We implement this greedy search approach in order to generate easy and hard tasks in the sampler class 'HardTaskSampler'. 

We note that the method implemented (i.e. greedy search) is incredibly expensive due to repeated sampling of the dataset. To mitigate this expense, we highly recommend using full dataset pre-extraction if using this sampler. 

### Random Tasks
In other few-shot works, evaluation has also been for the random N-Way K-Shot setting. We implement a task sampler which is capable of generating and solving these tasks. This task sampler 'AnyWayShotSampler' is present in the codebase but is currently not setup to function.


## Transfer Learning
This codebase now implements experiments for both [Pixels to Waveforms](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=_1R5aa8AAAAJ&citation_for_view=_1R5aa8AAAAJ:UeHWp8X0CEIC) and [On the Transferability of Large-Scale Self-Supervision...](https://arxiv.org/abs/2402.01274). These experiments generally included utilising a pre-trained model (either imagery or speech based) and applying it to the standard few-shot evaluation procedure as discussed above. More details about the types of experiments ran can be found in the respective papers.


### Image-Based Transfer Learning
Within the codebase we have a total of 107 usable pre-trained image models ready to go. These are split between supervised and self-supervised pre-training. 

#### Self-Supervised Model Setup
Supervised image-based models are taken from the TorchVision package and as such are accessible by default (no extra downloads needed). The self-supervised models however have to be downloaded. Steps to setup:
 - Download the model folder 'image_checkpoints' from [Google Drive](https://drive.google.com/drive/folders/1odfioT7V2Wc1xObUHWH8PE1jZgKB_0f0?usp=sharing)
 - Place folder of models wherever you like
 - Modify the model base paths within the files inside 'models/imagenet_pretrained'

#### Running an Experiment
Experiments are run very similarly to the baselines but with some key factors changed. 
 - Experiments for image-based models are run using 'imagenet_main.py'

A sample run may look something like:

```bash
python imagenet_main.py --model_name dino_vits14 --num_tasks 3000 --split test --rep_length 5 --classifier sklin --results_file simsiam --fine_tune False --n_way 5 --k_shot 1 --q_queries 5 --in_norm True --in_resize True
```

#### Additional Variables
Within the ImageNet experiments, we investigate additional variables named 'in_norm' (whether or not to apply ImageNet normalisation to incoming data) and 'in_resize' (wether or not to resize incoming spectrogram to standard ImageNet sizes). When running an experiment these can either be set to True or False (as can be seen above).



### SSL Speech Transfer Learning
We consider a total of 13 large-scale self-supervised speech models. Although there are more available within the SUPERB benchmark, we sub-select for consistent pre-training dataset (LS960). This is covered more in the paper.

#### Model Download
Included models run using a combination of configuration code (already included in teh checkpoint) and downloaded checkpoints (not already included). To setup the checkpoints:
 - Download the folder named 'checkpoints' from [here](https://drive.google.com/drive/folders/1odfioT7V2Wc1xObUHWH8PE1jZgKB_0f0?usp=sharing)
 - Place the folder in the 's3prl'
 - Should be all good

#### Running an Experiment
Experiments are run very similarly to the baselines but with some key factors changed. 
 - Experiments for speech-based models are run using 's3prl_main.py'

A sample run may look something like:

```bash
python .\s3prl_main.py  --model_name tera --num_tasks 10000 --split test --classifier sklin --results_file wavlm_5s_10000_sklin_test --dims 1 --in_channels 1 --rep_length 5 --fine_tune False
```