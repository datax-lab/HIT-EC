# HIT-EC: Trustworthy prediction of enzyme commission numbers using a hierarchical interpretable transformer
## Louis Dumontet†, So-Ra Han†, Tae-Jin Oh and Mingon Kang 
†Co-first authors: Louis Dumontet and So-Ra Han

## 1. Abstract 

**Motivation:** Accurate and trustworthy prediction of Enzyme Commission (EC) numbers is critical for understanding enzyme functions and their roles in biological processes. Despite the success of recently proposed deep learning based models, there remain limitations, such as low performance in underrepresented EC numbers, lack of learning strategy with incomplete annotations, and limited interpretability. To address these challenges, we propose a novel hierarchical interpretable transformer model, HIT-EC, for trustworthy EC number prediction. HIT-EC employs a four level transformer architecture that aligns with the hierarchical structure of EC numbers, and leverages both local and global dependencies within protein sequences for this multi-label classification task. We also propose a novel learning strategy to handle incomplete EC numbers. HIT-EC, as an evidential deep learning model, produces trustworthy predictions by providing domain-specific evidence through a biologically meaningful interpretation scheme.

**Results:** The predictive performance of HIT-EC was assessed by multiple experiments: cross-validation including underrepresented EC numbers, validation with external data, and species-based performance evaluation. HIT-EC showed statistically significant improvement in predictive performance when compared to the current state-of-the-art benchmark models. HIT-EC’s robust interpretability was further validated by identifying well-known conserved motifs and functionalregions in the CYP106A2 enzyme family. HIT-EC would be a robust, interpretable, and reliable solution for EC number prediction, with significant implications for enzymology, drug discovery, and metabolic engineering.

## 2. Requirements 

The manuscript results were obtained with: 

Python == 3.9.16  
Torch == 1.13.1  
PyTorch Lightning == 1.9.0  
Numpy == 1.23.5

Check the Demo notebook to get started. The model weights are avalaible in the first release 'Model weights'
