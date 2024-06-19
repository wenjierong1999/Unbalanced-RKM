'''
RLS sampling
Experiment on unbalanced MNIST012 dataset

candidate models:
vanilla Gen-RKM
RLS Gen-RKM (shared featuremap)
RLS Gen-RKM + umap (pretrained network as featuremap)
RLS Gen-RKM without umap (pretrained classifer as featuremap)

evaluation metrics:

Count of Generated minority modes : use a pre-trained classifier, predict labels for generated samples,
count the number of minority modes and compare it with the number of other modes.

KL divergence:
Compute KL divergence between classified label distribution of generated samples and a balanced label distribution.
Ideally, KL divergence should be close to zero which indicates a  balanced generation.

default unbalance ratio = 0.1


'''