# Use https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running
FROM nvcr.io/nvidia/pytorch:18.11-py3

RUN pip install gsutil
RUN pip install setproctitle
RUN conda install -y pandas
RUN conda install -c conda-forge scikit-image

RUN mkdir /workspace/job
COPY . /workspace/job/

ENTRYPOINT ["/workspace/job/docker_entrypoint.sh"]
