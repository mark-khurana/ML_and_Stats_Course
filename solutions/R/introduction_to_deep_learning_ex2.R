# =============================================================================
# Chapter 8b, Exercise 2: Architecture Matching
# For each clinical task, identify the best DL architecture, whether DL
# is likely to outperform gradient-boosted trees, and the reporting guideline.
# =============================================================================

# =============================================================================
# Task 1: Predicting 30-day mortality from 15 structured EHR variables
# =============================================================================
#
# (a) Architecture: Feedforward neural network (multi-layer perceptron) or,
#     better yet, gradient-boosted trees (XGBoost/LightGBM). With only 15
#     structured variables, a simple architecture suffices.
#
# (b) DL likely to outperform GBT? NO. This is classic tabular data with a
#     modest number of features. The Grinsztajn et al. (2022) NeurIPS
#     benchmark and subsequent clinical benchmarks consistently show that
#     tree-based models match or outperform neural networks on tabular data.
#     Logistic regression or XGBoost is the appropriate starting point.
#
# (c) Reporting guideline: TRIPOD+AI (Collins et al., BMJ 2024). This is a
#     standard clinical prediction model using structured data.

# =============================================================================
# Task 2: Classifying skin lesions as benign/malignant from dermoscopy images
# =============================================================================
#
# (a) Architecture: Convolutional Neural Network (CNN), specifically a
#     pretrained model like ResNet, EfficientNet, or Inception fine-tuned
#     on the dermoscopy images. A domain-specific foundation model could
#     also be used if available.
#
# (b) DL likely to outperform GBT? YES. Image data contains spatial
#     structure (edges, textures, shapes) that CNNs are specifically designed
#     to exploit. GBTs cannot process raw images and would require manual
#     feature extraction, which is inferior to learned CNN features. Esteva
#     et al. (Nature 2017) demonstrated dermatologist-level performance.
#
# (c) Reporting guideline: CLAIM (Checklist for AI in Medical Imaging,
#     Mongan et al., Radiology: AI 2020), supplemented by TRIPOD+AI.

# =============================================================================
# Task 3: Detecting atrial fibrillation from 12-lead ECG tracings
# =============================================================================
#
# (a) Architecture: Transformer-based model or temporal CNN. The chapter
#     notes that transformers are the current preferred architecture for ECG
#     data. Medformer (NeurIPS 2024) is specifically designed for medical
#     time series classification.
#
# (b) DL likely to outperform GBT? YES. ECG data is sequential with complex
#     temporal patterns. GBTs would require extensive manual feature
#     engineering (interval measurements, morphology features), whereas DL
#     can learn directly from the raw waveform. Hannun et al. (Nature
#     Medicine 2019) demonstrated cardiologist-level arrhythmia detection.
#
# (c) Reporting guideline: TRIPOD+AI for the prediction model, potentially
#     supplemented by CLAIM if imaging is involved (e.g., ECG images rather
#     than signal data).

# =============================================================================
# Task 4: Extracting medication names from unstructured discharge summaries
# =============================================================================
#
# (a) Architecture: Transformer-based language model, such as ClinicalBERT
#     or PubMedBERT for named entity recognition (NER). These pretrained
#     models understand medical vocabulary and can be fine-tuned for NER.
#
# (b) DL likely to outperform GBT? YES, decisively. This is a natural
#     language processing task. GBTs cannot process raw text meaningfully.
#     Transformer-based models understand context, synonyms, and medical
#     abbreviations. Rule-based and dictionary approaches are alternatives,
#     but modern NER with transformers is superior.
#
# (c) Reporting guideline: TRIPOD+AI. No specific imaging guideline applies.
#     MINIMAR (Hernandez-Boussard et al., JAMIA 2020) may also be relevant
#     as a minimum reporting standard.

# =============================================================================
# Task 5: Predicting length of stay from structured EHR + chest X-ray
# =============================================================================
#
# (a) Architecture: Multimodal fusion model. A CNN (e.g., pretrained
#     ResNet) processes the chest X-ray to extract image features. These
#     are concatenated with the structured EHR features and fed into a
#     combined prediction head (either a feedforward network or GBT on
#     the fused features).
#
# (b) DL likely to outperform GBT? PARTIALLY. The DL component is necessary
#     for the image. For the structured data alone, GBTs may be equal or
#     better. The optimal approach may be a hybrid: use a CNN to extract
#     image features, then combine those features with structured data in
#     a GBT. Recent work on multimodal fusion suggests this hybrid approach
#     can outperform either modality alone.
#
# (c) Reporting guideline: TRIPOD+AI for the overall prediction model,
#     supplemented by CLAIM for the imaging component. Both should be
#     addressed since the model involves medical imaging.

cat("This exercise is conceptual. See the comments in this file for the\n")
cat("complete answers to all five clinical tasks.\n")
