`markdown`
# 🎨 NeuralStyleTransferApp

A Flask web app that applies neural style transfer to images using a pre-trained VGG19 model. Users upload a content and style image, and the app generates and displays a styled image with an option to download it.

---

## 🚀 Features

* 🎨 **Neural Style Transfer**
* 📥 **Image Upload Support**
* ⏳ **Processing Indicator**
* 💾 **Download Option**

---

## 🛠️ Tech Stack

* **Frontend**: HTML5, CSS3, JavaScript
* **Backend**: Python 3, Flask
* **Libraries**:
  * [`torch`](https://pytorch.org/)
  * [`Flask`](https://pypi.org/project/Flask/)
  * [`PIL`](https://pillow.readthedocs.io/en/stable/)

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/NeuralStyleTransferApp.git
cd NeuralStyleTransferApp
````

### 2. Install Python Dependencies

```bash
pip install flask torch torchvision Pillow
```

---

## ▶️ Run the App

```bash
python app.py
```

Open your browser and go to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📂 Project Structure

```
NeuralStyleTransferApp/
│
├── static/
│   └── uploads/
│   └── output/
│
├── templates/
│   └── index.html
│
├── app.py
├── nst.py
├── README.md
```

---

## 📌 Usage

* Upload your **content image** and **style image**.
* Wait for the style transfer to complete.
* The styled image will be displayed.
* Click **Download** to save the styled image.

---

## 👨‍💻 Author

Anishkumar K
GitHub: [Anishkumar-1804](https://github.com/Anishkumar-1804)

---

## 📜 License

This project is open-source and free to use for educational or personal projects.

```
```
