# Use https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running
FROM nvcr.io/nvidia/pytorch:18.11-py3

COPY . /workspace/

ENTRYPOINT ["python", "-u", "/workspace/train.py"]
