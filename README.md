
# 🎤 Deepfake Audio Detection using Deep Learning

## 🚀 Project Overview

This project focuses on detecting **deepfake (AI-generated) audio** using **deep learning techniques**. With the rise of synthetic speech models, it has become essential to differentiate between real and fake audio to prevent misinformation and fraud.

Our approach involves **MFCC feature extraction**, followed by training a **CNN-based model** to classify real and fake voices.

🔗 **Colab Notebook**: [Deepfake Audio Detection](https://colab.research.google.com/drive/1J9yxm2HTQFSXV_CG7Fw5XKjgvzUGmLff?usp=sharing)

---

## 📌 Features

✅ Extracts MFCC (Mel-Frequency Cepstral Coefficients) from audio data  
✅ Uses **CNN-based models** for classification  
✅ Handles **overfitting** with dropout, batch normalization, and data augmentation  
✅ Trained on **real and deepfake audio datasets**  
✅ **Google Colab support** for efficient GPU-based training

---

## 🛠️ Tech Stack & Libraries Used

To ensure optimal performance, we leveraged the following tools:

| Category             | Libraries Used                                            |
| -------------------- | --------------------------------------------------------- |
| **Deep Learning**    | `TensorFlow`, `Keras`                                     |
| **Audio Processing** | `Librosa`, `SciPy`, `Soundfile`                           |
| **Data Handling**    | `NumPy`                                                   |
| **Visualization**    | `Matplotlib`, `Seaborn`                                   |
| **File Handling**    | `OS`, `Shutil`                                            |
| **Evaluation**       | `Scikit-learn` (classification reports, confusion matrix) |

🔹 **Install dependencies** using:

```bash
pip install tensorflow librosa numpy pandas matplotlib seaborn soundfile
```

---

## 🔍 Dataset & Preprocessing

We used a **real vs fake audio dataset**, where:

- **Real audio** is collected from genuine voice recordings
- **Fake audio** is generated using deepfake speech synthesis

### **Feature Extraction: MFCCs**

- **Why MFCCs?** Mel-Frequency Cepstral Coefficients (MFCCs) help extract spectral features of audio, making it easier for deep learning models to identify patterns.
- We used **Librosa** to extract **13 MFCCs per frame**, with `n_mfcc=13`.

```python
import librosa
import numpy as np

# Function to extract MFCC features
def extract_mfcc(file_path, max_pad_len=128):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs
```

---

## 🎯 Training Strategy

- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Binary Cross-Entropy (since we are doing binary classification)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Early Stopping**: Stops training if validation loss stops improving
- **Data Augmentation**: Pitch shifting, time stretching, adding background noise

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(train_features, train_labels,
                    epochs=20,
                    batch_size=32,
                    validation_data=(val_features, val_labels),
                    callbacks=[early_stopping])
```

---

## 📊 Results & Performance

- **Training Accuracy**: **~90%**
- **Validation Accuracy**: **~88%**
- **Test Accuracy**: **~72%**
- **Loss Reduction**: Successfully minimized overfitting with dropout layers and batch normalization.

### **Confusion Matrix**

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(test_features) > 0.5
cm = confusion_matrix(test_labels, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

---

## 💡 Challenges & How I Overcame Them

### 🚧 **1. Dataset Preprocessing**

🔹 Issue: Audio preprocessing and MFCC extraction were complex.  
✅ **Solution**: Used **Librosa** and ensured all features had uniform padding.

### 🚧 **2. Overfitting**

🔹 Issue: The model was overfitting on training data.  
✅ **Solution**: Implemented **dropout layers, data augmentation, and batch normalization**.

### 🚧 **3. Limited Computational Power**

🔹 Issue: Training deep learning models was resource-intensive.  
✅ **Solution**: Used **Google Colab’s free GPU acceleration** to speed up training.

---

## 📌 Future Improvements

🔹 Test on larger datasets for better generalization  
🔹 Experiment with **LSTM or Transformer models** for improved feature learning  
🔹 Deploy as a **real-time deepfake detection tool** using Flask or FastAPI

---

## 📢 Conclusion

This project was an exciting deep dive into **audio deepfake detection** using **CNNs and ResNet**. The results show promising accuracy, and with more data and fine-tuning, we can improve real-time deepfake detection in audio! 🚀

Feel free to **fork the repo, contribute, or provide suggestions!** 🙌

---

### ⭐ If you found this useful, please give it a **star**! ⭐

---

🚀 **#DeepLearning #AI #AudioProcessing #DeepfakeDetection #TensorFlow #MachineLearning**
