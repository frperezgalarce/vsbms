# vsbms
 
# Abstract 

Machine learning has achieved an important role in the automatic classification of variable stars, and several classifiers have been proposed over the last decade. These classifiers have achieved impressive performance in several astronomical catalogues. However, some scientific articles have also shown that the training data therein contain multiple sources of bias. Hence, the performance of those classifiers on objects not belonging to the training data is uncertain, potentially resulting in the selection of incorrect models. Besides, it gives rise to the deployment of misleading classifiers. An example of the latter is the creation of open-source labelled catalogues with biased predictions. In this paper, we develop a method based on an informative marginal likelihood to evaluate variable star classifiers. We collect deterministic rules that are based on physical descriptors of RR Lyrae stars, and then, to mitigate the biases, we introduce those rules into the marginal likelihood estimation. We perform experiments with a set of Bayesian Logistic Regressions, which are trained to classify RR Lyraes, and we found that our method outperforms traditional non-informative cross-validation strategies, even when penalized models are assessed. Our methodology provides a more rigorous alternative to assess machine learning models using astronomical knowledge. From this approach, applications to other classes of variable stars and algorithmic improvements can be developed.



# requirements
 
 pymc3 3.6
 
 numpy 1.15.4
 
 pandas 0.24.2
 
 sklearn 0.19.2
