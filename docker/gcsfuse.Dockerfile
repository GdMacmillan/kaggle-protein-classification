FROM pytorch/pytorch:0.4_cuda9_cudnn7

RUN conda install pandas scikit-learn scikit-image tqdm
RUN pip install setproctitle google-cloud-storage gsutil

RUN apt update && apt install rsync --yes

ENV GCSFUSE_REPO=gcsfuse-xenial
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
    lsb-release \
  && echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --yes gcsfuse \
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir /mnt/hpa-data
COPY . /home/human-protein-atlas

WORKDIR /home/human-protein-atlas/
