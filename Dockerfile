# Use https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running
FROM nvcr.io/nvidia/pytorch:18.11-py3

RUN pip install gsutil
RUN pip install setproctitle
RUN conda install -y pandas
RUN conda install -c conda-forge scikit-image


ENV GCSFUSE_REPO gcsfuse-stretch
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
  && echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --yes gcsfuse \
&& apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir /workspace/job
COPY . /workspace/job/

RUN mkdir /mnt/human-protein-atlas-data

ENV BOTO_CONFIG=job/.boto

ENTRYPOINT ["/workspace/job/docker_entrypoint.sh"]
