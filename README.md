# 🧬 HIT-EC: Hierarchical Interpretable Transformer for Enzyme Commission Number Prediction

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Paper](https://img.shields.io/badge/Nature%20Communications-Under%20Review-orange.svg)

---

## 📖 Abstract

Accurate and trustworthy prediction of Enzyme Commission (EC) numbers is critical for understanding enzyme functions and their roles in biological processes. 
Despite the success of recently proposed deep learning-based models, there remain limitations, such as low performance in underrepresented EC numbers, lack of learning strategy with incomplete annotations, and limited interpretability. To address these challenges, we propose a novel hierarchical interpretable transformer model, **HIT-EC**, for trustworthy EC number prediction. **HIT-EC** employs a four-level transformer architecture that aligns with the hierarchical structure of EC numbers, and leverages both local and global dependencies within protein sequences for this multi-label classification task. 
We also propose a novel learning strategy to handle samples associated with incomplete EC numbers. 
**HIT-EC**, as an evidential deep learning model, produces trustworthy predictions by providing domain-specific evidence through a biologically meaningful interpretation scheme. 
The predictive performance of HIT-EC was assessed by multiple experiments: a cross-validation with a large dataset, a validation with external data, and a species-based performance evaluation. 
**HIT-EC** showed statistically significant improvement in predictive performance when compared to the current state-of-the-art benchmark models. 
**HIT-EC**'s robust interpretability was further validated by identifying well-known conserved motifs and functional regions. 
**HIT-EC** would be a robust, interpretable, and reliable solution for EC number prediction, with significant implications for enzymology, drug discovery, and metabolic engineering.

---

## 🧰 Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/datax-lab/HIT-EC.git
cd HIT-EC
pip install -r requirements.txt
```
---

## 🌐 Online Model Predictions & Interpretations

You can use HIT-EC directly through our interactive web platform:

👉 **Website:** [https://enzymex.dataxlab.org/](https://enzymex.dataxlab.org/)

The website allows you to:

* Upload FASTA sequences
* Run HIT-EC predictions
* Visualize contribution scores

---

## 🚀 Quick Demo

Pretrained model weights are released under **version v2.0.0** on GitHub.  
You can use the **demo notebook** to predict and interpret the model's predictions.

---

## 🧪 Training Your Own Model

Use the training scripts `train/infer_train.py` and `train/inter_train.py` to train HIT-EC from scratch on your curated enzyme dataset. 

---

## 📁 Repository Structure

```
HIT-EC/
│
├── model/              # Transformer
├── data/               # Datasets used in the paper
├── training/           # Training pipelines
├── utils/              # Label encoder and tokenizer
├── Demo.ipynb          # Demo notebook
└── requirements.txt    # Necessary packages
```

---

## 📦 Pretrained Model Weights

- **Release:** [v2.0.0](https://github.com/datax-lab/HIT-EC/releases/tag/v2.0.0)  
- **File:** `model.ckpt`  
- **Description:** Fully trained model with interpretability heads enabled.

---

## ⏱ Runtime and Performance

The demo notebook runs end-to-end in a few seconds on any modern CPU or GPU.
It automatically loads the pretrained model (v2.0.0) and performs both prediction and interpretation on example enzyme sequences.

Full model training can take several hours to multiple days, depending on your hardware setup.

Multi-GPU distributed training is supported for large-scale datasets.

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

**Contact:** louis.dumontet@unlv.edu  
**Maintained by:** DataX-Lab, UNLV
