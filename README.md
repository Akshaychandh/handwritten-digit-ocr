# 🔢 Handwritten Digit OCR

A deep learning pipeline to extract handwritten number sequences from images.  
Trained on **MNIST** using a custom **CNN in PyTorch** — no external OCR APIs used.

---

## 📁 Repo Structure

```
handwritten-digit-ocr/
├── src/
│   └── ocr_digits.py        ← Main OCR script
├── output/
│   └── digit_cnn.pth        ← Trained model weights (after training)
├── report/
│   └── OCR_Report.docx      ← Full technical report
├── Train_DigitOCR.ipynb     ← Google Colab training notebook
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/handwritten-digit-ocr.git
cd handwritten-digit-ocr
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train on Google Colab *(recommended — free GPU)*
- Open `Train_DigitOCR.ipynb` in [Google Colab](https://colab.research.google.com/)
- Set **Runtime → T4 GPU**
- Run all cells
- Download `digit_cnn.pth` → place it in `output/`

### 4. Run OCR on your images
```bash
# Single image
python src/ocr_digits.py images/sample.jpg --model output/digit_cnn.pth

# Folder of images + debug visualisation
python src/ocr_digits.py images/ --model output/digit_cnn.pth --debug
```

---

## 🧠 Model Architecture

| Layer        | Config              | Output       |
|--------------|---------------------|--------------|
| Conv1        | 3×3, 32 filters     | 32 × 28 × 28 |
| Conv2        | 3×3, 64 filters     | 64 × 28 × 28 |
| MaxPool + Dropout | 2×2, 25%      | 64 × 14 × 14 |
| Conv3        | 3×3, 128 filters    | 128 × 14 × 14|
| MaxPool + Dropout | 2×2, 25%      | 128 × 7 × 7  |
| FC1          | 6272 → 256, Dropout | 256          |
| FC2 (Output) | 256 → 10            | 10 classes   |

**Optimizer:** Adam | **LR:** 1e-3 → StepLR | **Epochs:** 10  
**Test Accuracy on MNIST:** >99.2%

---

## 🔄 Pipeline

```
Input Image
    ↓ Grayscale + Gaussian Blur
    ↓ Adaptive Thresholding (handles uneven lighting)
    ↓ Morphological Close (fill digit gaps)
    ↓ Contour Detection + Line Grouping
    ↓ 28×28 Crop per digit
    ↓ CNN Inference
    ↓ JSON Output
```

---

## 📊 Output Format

Results are saved to `output/results.json`:
```json
[
  {
    "image": "receipt.jpg",
    "lines": [
      { "line": 1, "text": "7012233066", "digits": [
          { "digit": 7, "confidence": 0.99, "bbox": [10, 5, 12, 20] }
        ]
      }
    ],
    "full_text": "7012233066 01534 23.19"
  }
]
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | CNN training & inference |
| torchvision | MNIST dataset + transforms |
| OpenCV | Image preprocessing & segmentation |
| Pillow | Image loading |
| NumPy | Array operations |

---

## 📄 Report

See [`report/OCR_Report.docx`](report/OCR_Report.docx) for the full technical report covering architecture, results, challenges, and findings.

---

## 📝 Assignment Constraints Met

- ✅ No external OCR APIs (no Google Vision, Tesseract API, AWS Textract)
- ✅ Open-source tools only
- ✅ Deep learning approach (CNN)
- ✅ MNIST dataset used for training
- ✅ Code + Report submitted
