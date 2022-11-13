# Mechanistic Mode Connectivity


Codebase for the paper "Mechanistic Mode Connectivity".

Contains code for generating and training models on synthetic datasets, finding and evaluating connectivity paths between two minimizers, running counterfactual evaluation, and executing several fine-tuning methods.

## Requirements

The code requires:

* Python 3.6 or higher

* Pytorch 1.8 or higher

Dependencies can be installed via the following command:

```setup
pip -r install requirements.txt
```


## Example execution commands

Training VGG-13 models on CIFAR-10 with box cues dataset with 80% of data containing synthetic cues

```trainer
python trainer.py --model=VGG-13 --base_dataset=CIFAR10 --cue_type=box --cue_proportion=0.8 --n_epochs=100
```


Evaluating linear paths between VGG-13 models on different counterfactuals of CIFAR-10 with box cues dataset. Here, a model trained without cues and one trained with 80% of data containing cues are being evaluated.

```evalpath
python train_eval_mc.py --perform=eval_path --cue_type=box --base_dataset=CIFAR10 --cue_proportion=0.8 --connectivity_pattern=LMC --model=VGG-13 --eval_data=test --n_interpolations=10
```


Finding midpoints for identifying a quadratic path between VGG-13 models on CIFAR-10 with box cues dataset.

```midpoints
python train_eval_mc.py --model=VGG-13 --perform=train_midpoint --cue_type=box --id_data=cue --base_dataset=CIFAR10 --cue_proportion=0.8
```


Mechanistic fine-tuning using CBFT of a VGG-13 model trained on CIFAR-10 with box cues dataset that contained 80% samples with synthetic cues.

```mechft
python mech_fine_tuning.py --model=VGG-13 --base_dataset=CIFAR-10 --cue_model_path=path_to_model --cue_type=box --cue_proportion=0.8 --n_epochs=20 --ft_method=CBFT --n_clean=2500 --n_cue=47500
```



## Organization

* **trainer.py**: Train VGG-13 / ResNet-18 models on different synthetic / natural datasets

* **train_eval_mc.py**: Main execution file for evaluating accuracy / loss along points on a path and training midpoints for a quadratic path

* **mech_fine_tuning.py**: Implementations for different fine-tuning methods

* **syndata.py**: Implementation for synthetic / natural datasets; also contains counterfactual generators for "rand. cue" and "rand. image" protocols from the paper

* **mode_connect.py**: Contains functions for finding parameters at a specific point on a path; training midpoints for quadratic paths; running counterfactual evaluations on a path

* **find_permutation.py**: Contains functions for finding permutations that maximally match two models in activations

* **models.py**: Model definitions (VGG-13 / ResNet-18)

* **utils.py**: Test evaluation function / Learning rate scheduler

