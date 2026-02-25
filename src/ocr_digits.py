"""
Handwritten Digit OCR — Deep Learning Approach
================================================
Uses a CNN trained on MNIST to extract digit sequences from images.
Dependencies: torch, torchvision, opencv-python, Pillow, numpy

Install:
    pip install torch torchvision opencv-python Pillow numpy
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# 1. CNN Model Definition
# ─────────────────────────────────────────────────────────────────────────────

class DigitCNN(nn.Module):
    """
    Convolutional Neural Network for handwritten digit recognition.
    Architecture inspired by LeNet-5 with added Dropout for regularisation.
    Input: 1×28×28 grayscale image
    Output: 10-class softmax (digits 0-9)
    """
    def __init__(self):
        super(DigitCNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28×28 → 28×28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28×28 → 14×14 (after pool)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 14×14 → 7×7 (after pool)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)

        # Fully connected
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2. Training on MNIST
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model_path="digit_cnn.pth", epochs=10, batch_size=128):
    """Train the CNN on MNIST and save the weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using device: {device}")

    # Data transforms — normalise to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download MNIST
    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False, num_workers=2)

    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # — Train
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # — Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100.0 * correct / total
        scheduler.step()
        print(f"  Epoch {epoch:02d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            print(f"  → Model saved (best acc: {best_acc:.2f}%)")

    print(f"\n[Training complete] Best accuracy: {best_acc:.2f}%")
    return model_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. Image Preprocessing & Digit Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img_path):
    """
    Load an image and return a binary (thresholded) version ready for
    contour-based digit segmentation.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding handles uneven lighting (receipt scans, shadows)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=10
    )

    # Morphological operations to close gaps inside digits
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return img, gray, binary


def segment_digits(binary):
    """
    Find bounding boxes for individual digit regions via contour detection.
    Returns a list of (x, y, w, h) sorted left→right, top→bottom.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    h_img, w_img = binary.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter noise: ignore very small or very large blobs
        if w < 5 or h < 5:
            continue
        if w > w_img * 0.9 or h > h_img * 0.9:
            continue
        # Aspect ratio filter: digits are roughly 0.2 – 2.0 wide/tall
        ar = w / float(h)
        if ar > 2.5 or ar < 0.1:
            continue

        boxes.append((x, y, w, h))

    # Sort top→bottom, left→right (group into lines first)
    if not boxes:
        return boxes

    # Simple line grouping by y-centre proximity
    boxes = sorted(boxes, key=lambda b: b[1])
    lines = []
    current_line = [boxes[0]]
    for b in boxes[1:]:
        prev_y = current_line[-1][1] + current_line[-1][3] // 2
        cur_y  = b[1] + b[3] // 2
        if abs(cur_y - prev_y) < 20:
            current_line.append(b)
        else:
            lines.append(sorted(current_line, key=lambda x: x[0]))
            current_line = [b]
    lines.append(sorted(current_line, key=lambda x: x[0]))

    return lines  # List of lines, each line is a list of (x,y,w,h)


def crop_digit(gray, x, y, w, h, padding=4, size=28):
    """Crop a digit region, add padding, resize to 28×28."""
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(gray.shape[1], x + w + padding)
    y1 = min(gray.shape[0], y + h + padding)

    roi = gray[y0:y1, x0:x1]
    roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
    return roi


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path="digit_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def predict_digit(model, device, gray_crop):
    """Run CNN inference on a single 28×28 grayscale crop."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Invert if background is light (MNIST has white digit on black)
    if gray_crop.mean() > 127:
        gray_crop = 255 - gray_crop

    pil_img = Image.fromarray(gray_crop).convert("L")
    tensor = transform(pil_img).unsqueeze(0).to(device)  # 1×1×28×28

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return predicted.item(), confidence.item()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_image(img_path, model, device, debug=False, output_dir=None):
    """
    Full pipeline: load → preprocess → segment → classify → return results.
    """
    print(f"\n[Processing] {img_path}")
    img, gray, binary = preprocess_image(img_path)

    lines = segment_digits(binary)
    if not lines:
        print("  No digits found.")
        return {"image": img_path, "lines": [], "full_text": ""}

    results = []
    annotated = img.copy()

    for line_idx, line_boxes in enumerate(lines):
        line_digits = []
        for (x, y, w, h) in line_boxes:
            crop = crop_digit(gray, x, y, w, h)
            digit, conf = predict_digit(model, device, crop)

            line_digits.append({
                "digit": digit,
                "confidence": round(conf, 4),
                "bbox": [x, y, w, h]
            })

            if debug:
                color = (0, 255, 0) if conf > 0.8 else (0, 165, 255)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated, str(digit), (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        line_str = "".join(str(d["digit"]) for d in line_digits)
        print(f"  Line {line_idx + 1}: {line_str}")
        results.append({"line": line_idx + 1, "digits": line_digits, "text": line_str})

    full_text = " ".join(r["text"] for r in results)

    if debug and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{base}_annotated.jpg"), annotated)
        cv2.imwrite(os.path.join(output_dir, f"{base}_binary.jpg"), binary)
        print(f"  Debug images saved to {output_dir}/")

    return {"image": img_path, "lines": results, "full_text": full_text}


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Handwritten Digit OCR using CNN trained on MNIST"
    )
    parser.add_argument("images", nargs="+", help="Path(s) to input image(s)")
    parser.add_argument("--model",  default="digit_cnn.pth", help="Path to saved model weights")
    parser.add_argument("--train",  action="store_true",     help="Train model before inference")
    parser.add_argument("--epochs", type=int, default=10,    help="Training epochs (default: 10)")
    parser.add_argument("--debug",  action="store_true",     help="Save annotated debug images")
    parser.add_argument("--output", default="output",        help="Output directory for results")
    args = parser.parse_args()

    # Train if requested or no saved model exists
    if args.train or not os.path.exists(args.model):
        print("[Step 1/2] Training model on MNIST …")
        train_model(model_path=args.model, epochs=args.epochs)
    else:
        print(f"[Step 1/2] Loading pre-trained model from '{args.model}' …")

    print("\n[Step 2/2] Running OCR on input images …")
    model, device = load_model(args.model)

    all_results = []
    for img_path in args.images:
        result = process_image(img_path, model, device,
                               debug=args.debug, output_dir=args.output)
        all_results.append(result)

    # Save results to JSON
    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Results saved to {json_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for r in all_results:
        print(f"  Image : {r['image']}")
        print(f"  Output: {r['full_text']}")
        print()


if __name__ == "__main__":
    main()
