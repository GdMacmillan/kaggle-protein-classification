if [ -z $REPO_ROOT ]; then
    echo "set the \$REPO_ROOT (full path of the root of this repository)"
    exit 1
fi

PLATFORM=gcp
KUBEFLOW_REPO=/opt/kubeflow
KUBEFLOW_VERSION=master
KUBEFLOW_KS_DIR=${REPO_ROOT}/hpakf/ks_app
KUBEFLOW_DOCKER_REGISTRY=
K8S_NAMESPACE=kubeflow
KUBEFLOW_CLOUD=gke
PROJECT=optfit-kaggle
DEPLOYMENT_NAME=hpakf
KUBEFLOW_DM_DIR=${REPO_ROOT}/hpakf/gcp_config
KUBEFLOW_SECRETS_DIR=${REPO_ROOT}/hpakf/secrets
KUBEFLOW_K8S_MANIFESTS_DIR=${REPO_ROOT}/hpakf/k8s_specs
KUBEFLOW_K8S_CONTEXT=hpakf
ZONE=us-east1-d
EMAIL=barton@optfit.ai
KUBEFLOW_IP_NAME=hpakf-ip
KUBEFLOW_ENDPOINT_NAME=hpakf
KUBEFLOW_HOSTNAME=hpakf.endpoints.optfit-kaggle.cloud.goog
CONFIG_FILE=cluster-kubeflow.yaml
PROJECT_NUMBER=33880501045
CLUSTER_VERSION=1.10.9-gke.0
