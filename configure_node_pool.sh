gcloud container node-pools create gpu-pool-k80 \
--accelerator type=nvidia-tesla-k80,count=2 --zone us-east1-d \
--cluster hpakf --num-nodes 1 --min-nodes 1 --max-nodes 5 \
--enable-autoscaling
