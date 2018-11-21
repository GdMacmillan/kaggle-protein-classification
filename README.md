Kaggle Human Protein Atlas Image Competition
====
### Using Kubeflow and Kubernetes to distribute training of a classifier for sub-cellular protein patterns in human cells
----

This project is about the classification of human proteins in images and distributing the training of a neural net using Kubernetes/Kubeflow.

The task is multi-label classification so we have to predict which labels are relevant to the picture:

Y ⊆ { Peroxisomes, Endosomes, Lysosomes, Intermediate filaments, Actin filaments, Focal adhesion sites, Microtubules... }

i.e., each instance can have multiple labels instead of a single one!
