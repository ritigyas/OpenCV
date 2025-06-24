## 🤟 Real-Time Sign Language Detection using Python, OpenCV & MediaPipe

This project uses machine learning to detect and classify hand gestures from webcam input in real time. Built with Python, OpenCV, and MediaPipe, it recognizes five custom gestures including “hello”, “yes”, “no”, “I love you”, and a personalized sign “ritigya”.




### 🚀 Features

* ✅ Real-time gesture recognition via webcam
* ✅ Hand landmark detection using MediaPipe
* ✅ Random Forest classifier trained on custom dataset
* ✅ Balanced dataset with cross-validation for accuracy
* ✅ Supports 5 gestures and easily extendable

---

### 🧠 Technologies Used

* **Python 3.x**
* **OpenCV** — for real-time video and visualization
* **MediaPipe** — for hand landmark detection
* **Scikit-learn** — Random Forest classification
* **NumPy, Pickle** — data handling and storage

---

### 📁 Project Structure

```
📦 sign-language-detector/
├── collect_imgs.py         # Script to collect training images
├── create_dataset.py       # Extracts 21 hand landmarks from each image
├── train_classifier.py     # Trains a Random Forest classifier
├── inference.py            # Performs live predictions using webcam
├── data/                   # Directory containing class-wise images
├── data.pickle             # Saved features and labels
├── model.p                 # Trained classifier
```

---

### 🔧 Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/ritigyas/Real_Time_signLanguage_detection.git
   cd Real_Time_signLanguage_detection
   ```

2. Install the requirements:

   ```bash
   pip install opencv-python mediapipe scikit-learn numpy
   ```

3. Collect images:

   ```bash
   python collect_imgs.py
   ```

4. Create dataset:

   ```bash
   python create_dataset.py
   ```

5. Train model:

   ```bash
   python train_classifier.py
   ```

6. Run inference:

   ```bash
   python inference.py
   ```

---

### 📊 Results

* Achieved over **99% accuracy** on test set
* Used **5-fold cross-validation** for model generalization
* Live predictions are stable and responsive

---




---



