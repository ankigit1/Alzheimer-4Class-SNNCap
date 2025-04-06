# 🧠 Alzheimer’s Stage Classification Using Siamese Capsule Network (SNNCap)

This repository contains the codebase for our research project titled:

> **Introducing Siamese Capsule Network for Alzheimer’s Diagnosis via MRI Analysis**

We present a novel deep learning approach combining Siamese Networks with Capsule Networks for accurate classification of Alzheimer's stages from brain MRI scans.

---

## 📂 Folder Structure
```
├── main.py                  # Entrypoint to train & validate the model
├── inference.py             # Script for single image inference using saved checkpoint
├── train.py                 # Training loop logic
├── validate.py              # Validation and evaluation code
├── utils.py                 # Utility functions
├── models/
│   └── capsule_network.py   # CapsuleNet + Siamese wrapper architecture
├── checkpoints/             # <PLACEHOLDER> For storing trained weights (.pth)
├── embeddings/              # <PLACEHOLDER> For saved reference embeddings (.pt)
└── README.md                # Project documentation (this file)
```

---

## 📌 Highlights
- Developed a **Siamese Capsule Network** architecture that leverages spatial hierarchies in MRIs.
- Used a pairwise comparison loss to train the model to learn distances between class embeddings.
- Introduced **reference-based validation** using 5 representative images per class.
- Achieved **97.47% accuracy** on 4-class Alzheimer’s stage classification (NonDemented, VeryMildDemented, MildDemented, ModerateDemented).
- Evaluated on the **Mendeley Alzheimer’s MRI dataset**.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ankigit1/Alzheimer-4Class-SNNCap
cd Alzheimer-4Class-SNNCap
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place the dataset [(Mendeley MRI Alzheimer)](https://data.mendeley.com/datasets/ch87yswbz4/1) in this folder structure:
```
/dataset/
├── NonDemented/
├── VeryMildDemented/
├── MildDemented/
└── ModerateDemented/
```
Update the path in `main.py` accordingly.

### 4. Start training
```bash
python main.py
```

This will train the model and also save the best checkpoint and reference embeddings.

---

## 🧪 Single Image Inference
To test on a single image using saved embeddings:
```bash
python inference.py --image_path path/to/image.png
```
Make sure the following files exist:
- Trained model checkpoint: `checkpoints/checkpoint.pth`
- Reference embeddings file: `embeddings/reference_embeddings.pt`

---

## 📁 GDrive Links (for large files)
- ✅ [Checkpoint (.pth)](https://drive.google.com/file/d/14br_vhqeJ4HoeU7qrIvFTcUyVtaBSm7S/view?usp=sharing)
- ✅ [Reference Embeddings (.pt)](https://drive.google.com/file/d/1i7f58FWk-jFJ_PVgCJD-MZw5zDnKpQ46/view?usp=sharing)

---

## 📊 Results
| Metric       | Value   |
|--------------|---------|
| Accuracy     | 97.47%  |
| F1 Score     | 98.74%  |
| Precision    | 98.75%  |
| Recall       | 98.76%  |

Confusion Matrix and ROC Curves can be found in the paper/report.


```

## 🤝 Contact
For questions, please contact: [Ankit Garg](mailto:ankitgarg5745@gmail.com)

---

© 2025 Ankit Garg | M.Tech Data Analytics | NIT Jalandhar
