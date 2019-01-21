gcloud container node-pools create gpu-pool \
--accelerator type=nvidia-tesla-k80,count=2 --zone us-east1-d \
--cluster hpakf --num-nodes 1  --min-nodes 0 --max-nodes 5 \
--enable-autoscaling
