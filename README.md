# Ambiguous Medical Image Segmentation using Flow Matching

Accurately segmenting medical images amidst annotation variability among multiple experts, characterized by aleatoric uncertainty, remains a significant challenge. A single deterministic segmentation map often fails to capture the inherent variability in such scenarios. Instead, generating a diverse set of plausible segmentation hypotheses is more informative and useful. 

To address this, we introduce **MedFM**, a generative model for medical image segmentation based on **flow matching**. Our image-guided flow matching model captures segmentation ambiguity by learning plausible variations in expert annotations. It also enables exact likelihood computation for arbitrary segmentation masks using **Hutchinson’s trace estimation**, allowing quantitative validation of segmentation mask compatibility with the input image. 

Experimental results on the **LIDC-IDRI dataset**, a benchmark for ambiguous medical image segmentation, show that **MedFM** outperforms existing methods in accuracy and uncertainty representation.

## Results

| Method            | GED² ↓  | HM-IoU ↑ |
|-------------------|---------|----------|
| PhiSeg [1]        | 0.262   | 0.586    |
| CIMD [2]          | 0.321   | 0.542    |
| MoSE [3]          | 0.257   | 0.583    |
| **MedFM (ours)**  | **0.202** | **0.610** |

**Table:** Quantitative results on the LIDC-IDRI dataset. Sixteen segmentation masks were generated per image. Lower GED² indicates closer alignment with the ground truth segmentation distribution, while higher HM-IoU reflects improved segmentation accuracy.

## References
1. [PHiSeg: Capturing Uncertainty in Medical Image Segmentation](https://arxiv.org/abs/1906.04045)
2. [Ambiguous Medical Image Segmentation using Diffusion Models](https://arxiv.org/abs/2304.04745)
3. [Modeling Multimodal Aleatoric Uncertainty in Segmentation with Mixture of Stochastic Experts](https://arxiv.org/abs/2212.07328)
