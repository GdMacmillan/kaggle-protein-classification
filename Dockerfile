# Use https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running
FROM nvcr.io/nvidia/pytorch:18.11-py3

RUN mkdir /workspace/job
COPY . /workspace/job/

ENV BOTO_CONFIG=job/.boto

ENTRYPOINT ["job/docker_entrypoint.sh"]
