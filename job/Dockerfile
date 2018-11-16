# Use https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running
FROM nvcr.io/nvidia/pytorch:18.11-py3

ADD ../kaggle-protein-classification/data /workspace/data
ADD ../kaggle-protein-classification/src /workspace/src
ADD ../kaggle-protein-classification/__init__.py /workspace/__init__.py
ADD ../kaggle-protein-classification/train.py /workspace/train.py
ADD ../kaggle-protein-classification/test.py /workspace/test.py

ENTRYPOINT ["python", "-u", "/workspace/kaggle-protein-classification/test.py"]
