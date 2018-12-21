Kaggle Human Protein Atlas Image Competition
====

In development...

#### Using Kubeflow and Kubernetes to distribute training of a classifier for sub-cellular protein patterns in human cells
----

## Introduction

This project is about the classification of human proteins in images and distributing the training of a neural net using Kubernetes/Kubeflow.

The task is multi-label classification so we have to predict which labels are relevant to the picture:

Y ⊆ { Peroxisomes, Endosomes, Lysosomes, Intermediate filaments, Actin filaments, Focal adhesion sites, Microtubules... }

i.e., each instance can have multiple labels instead of a single one!

### Table of Contents
* [Project Scope](#h1)
* [Data Engineering](#h2)

## <a id="h1"></a> Project Scope
#### This project sets out to accomplish two things:
*   Create a stable cloud native training application for quickly iterating and guiding the development of classification models
*   Define a classifier to recognize patterns in sub-cellular protein image data

## <a id="h2"></a> Engineering
Engineering goals for this project are to safely deploy a full featured, open source and end-to-end model training pipeline which can successfully leverage recent advances in PyTorch, Kubernetes, Docker and GPU's

In addition to the benefits of Kubernetes/Docker for portability and managing the environment, we look to Kubeflow for simplifying the workflow of deploying our training application to cloud infrastructure.

### Kubeflow

See [ an introduction to kubeflow on Google Kubernetes engine to get started](https://codelabs.developers.google.com/codelabs/kubeflow-introduction/index.html). Kubeflow usees ksonnet to help manage deployment.

<\someone describe the building of the Kubeflow layer\\>

We describe two Kubeflow components we used in our project:
*   Jupyterhub
*   Pytorch-operator

#### Jupyterhub

One of the core components of Kubeflow is Jupyterhub for creating interactive notebook environments.

To configure a notebook, once the kubeflow application is deployed, go to `https://<cluster-name\>.endpoints.<project-name\>.cloud.goog`

Go to the Jupyterhub pane and click start my server. This will take you into the spawner options. Optionally, you can provide an image stored in either a public or private container registry. To use our public repo specify gcr.io/optfit-kaggle/jupyterhub-k8s:latest as the image. The default image will have several libraries including Tensorflow and Numpy. CPU, Memory and Extra resources can be specified as well before clicking spawn to create the notebook server.

#### Pytorch-operator

A component that is still very early on in the development cycle is pytorch-operator.

Anyone with privs on the cluster can run. Do `git pull` latest code then run:

`docker build -t gcr.io/optfit-kaggle/human-protein-atlas .`

`docker push gcr.io/optfit-kaggle/human-protein-atlas:latest`

`kubectl -n kubeflow create -f pytorch_job_hpa.yaml``
**If you receive a job already exists error trying to run a new pytorch job and no one else is trying to run the job in the same namespace, you must delete it from the pytorch jobs inventory on kubernetes in the namespace in which you are trying to run the job as there is no way to "rerun" a job.
 ex. `kubectl --namespace=kubeflow delete pytorchjob pytorch-human-protein-atlas`

to ssh to pod it’s training run:

`kubectl exec -it pytorch-human-protein-atlas-master-0 -- /bin/bash`

to see logs:

`kubectl -n kubeflow logs -f pytorch-human-protein-atlas-master-0`
