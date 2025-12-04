# Sports Image Classification using CNN | Deep Learning Project

This project is a deep-learning based **Sports Image Classification System** that uses a **Convolutional Neural Network (CNN)** to classify images into different sports categories.  
The system identifies the sport present in an image and displays its confidence score through an interactive **Streamlit Web Application**.

---

## Project Objective
To build a robust system that classifies sports images automatically using deep learning techniques, providing real-time predictions for practical use in sports analytics and media content management.

---

##  Sports Categories
The model is trained to classify images into the following **7 sports**:

- Badminton 
- Cricket 
- Karate 
- Soccer 
- Swimming 
- Tennis �
- Wrestling 

---

##  Dataset Download

The dataset and trained model files can be downloaded from Google Drive:

 **Dataset & Model Link**  
https://drive.google.com/drive/folders/1WBZUzedxZ5WhRr8QJHAixq9oRL7_6cXg?usp=drive_link

### Extract the downloaded files and move the dataset to:

cnn-cifar10-classification/
└── data/
├── train/
└── val/


### Place the model file in:

cnn-cifar10-classification/
└── artifacts/
└── best_model.h5


---

##  Model Architecture (CNN Overview)

| Layer Type | Description |
|------------|-------------|
| Conv2D + MaxPooling | Feature extraction from images |
| Conv2D + MaxPooling | Mid-level spatial learning |
| Conv2D + MaxPooling | High-level pattern recognition |
| Flatten | Convert feature maps to vector |
| Dense (256) | Fully connected learning layer |
| Dropout | Prevents overfitting |
| Dense (Softmax – 7 units) | Predicts final sport class |

### Training Parameters

| Parameter | Value |
|----------|--------|
| Image Size | 128 × 128 × 3 |
| Batch Size | 32 |
| Epochs | 20 |
| Loss | Categorical Crossentropy |
| Optimizer | Adam |


---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Web App | Streamlit |
| Data Processing | NumPy, Pillow, Pandas |
| Visualization | Matplotlib |
| Version Control | Git & GitHub |

---

## 🛠 Installation & Setup

STEP 1

### Clone the Repository
use the below link

git clone https://github.com/DevarshSabu/cnn-cifar10-classification.git
cd cnn-cifar10-classification

STEP 2

python -m venv venv
venv\Scripts\activate    

STEP 3

pip install -r requirements.txt

STEP 4

python src/train.py

STEP 5

python src/evaluate.py

STEP 6

streamlit run app/streamlit_app.py

FOLDER STRUCTURE

cnn-cifar10-classification/
│── app/
│   └── streamlit_app.py
│── artifacts/
│   └── best_model.h5
│── data/
│   ├── train/
│   └── val/
│── src/
│   ├── data_prep.py
│   ├── train.py
│   └── evaluate.py
│── .gitignore
│── README.md



