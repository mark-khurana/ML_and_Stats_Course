# =============================================================================
# Chapter 8b, Exercise 2: Architecture Matching
# For each clinical task, identify the best DL architecture, whether DL
# is likely to outperform gradient-boosted trees, and the reporting guideline.
# =============================================================================

# =============================================================================
# Task 1: Predicting 30-day mortality from 15 structured EHR variables
# =============================================================================
#
# (a) Architecture: Feedforward neural network (MLP) or, better yet,
#     gradient-boosted trees (XGBoost/LightGBM). With only 15 structured
#     variables, a simple architecture suffices.
#
# (b) DL likely to outperform GBT? NO. This is classic tabular data with a
#     modest number of features. The Grinsztajn et al. (2022) NeurIPS
#     benchmark and subsequent clinical benchmarks consistently show that
#     tree-based models match or outperform neural networks on tabular data.
#
# (c) Reporting guideline: TRIPOD+AI (Collins et al., BMJ 2024). Standard
#     clinical prediction model using structured data.

# =============================================================================
# Task 2: Classifying skin lesions as benign/malignant from dermoscopy images
# =============================================================================
#
# (a) Architecture: CNN (e.g., pretrained ResNet, EfficientNet, or Inception
#     fine-tuned on dermoscopy images). A domain-specific foundation model
#     could also be used if available.
#
# (b) DL likely to outperform GBT? YES. Image data contains spatial
#     structure that CNNs exploit. GBTs cannot process raw images. Esteva
#     et al. (Nature 2017) demonstrated dermatologist-level performance.
#
# (c) Reporting guideline: CLAIM (Mongan et al., Radiology: AI 2020),
#     supplemented by TRIPOD+AI.

# =============================================================================
# Task 3: Detecting atrial fibrillation from 12-lead ECG tracings
# =============================================================================
#
# (a) Architecture: Transformer-based model or temporal CNN. Transformers
#     are the current preferred architecture for ECG data. Medformer
#     (NeurIPS 2024) is designed for medical time series classification.
#
# (b) DL likely to outperform GBT? YES. ECG data is sequential with complex
#     temporal patterns. DL can learn directly from raw waveforms, whereas
#     GBTs require extensive manual feature engineering.
#
# (c) Reporting guideline: TRIPOD+AI, potentially supplemented by CLAIM if
#     ECG images rather than signal data are used.

# =============================================================================
# Task 4: Extracting medication names from unstructured discharge summaries
# =============================================================================
#
# (a) Architecture: Transformer-based language model (ClinicalBERT or
#     PubMedBERT) for named entity recognition (NER). These models
#     understand medical vocabulary and context.
#
# (b) DL likely to outperform GBT? YES, decisively. This is an NLP task.
#     GBTs cannot process raw text meaningfully. Transformer-based NER
#     models understand context, synonyms, and medical abbreviations.
#
# (c) Reporting guideline: TRIPOD+AI. MINIMAR (Hernandez-Boussard et al.,
#     JAMIA 2020) also applicable as minimum reporting standard.

# =============================================================================
# Task 5: Predicting length of stay from structured EHR + chest X-ray
# =============================================================================
#
# (a) Architecture: Multimodal fusion model. A CNN (pretrained ResNet)
#     extracts image features from the chest X-ray. These are concatenated
#     with structured EHR features and fed into a combined prediction head.
#
# (b) DL likely to outperform GBT? PARTIALLY. DL is necessary for the image
#     component. For structured data alone, GBTs may be equal or better.
#     A hybrid approach (CNN for image features, then GBT on fused features)
#     may be optimal.
#
# (c) Reporting guideline: TRIPOD+AI for the prediction model, supplemented
#     by CLAIM for the imaging component.

print("This exercise is conceptual. See the comments in this file for the")
print("complete answers to all five clinical tasks.")
