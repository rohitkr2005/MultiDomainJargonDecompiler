# 🔍 Multi-Domain Jargon Decompiler

An AI-powered document analyzer designed to simplify complex legal contracts and medical reports into plain, easy-to-understand English. 

Built with **Python**, **Streamlit**, and **OpenRouter AI**, this tool uses OCR (Optical Character Recognition) to extract text from images or PDFs and "decompiles" jargon into layman's terms.

---

## ✨ Features

- **📑 Legal Analysis:** Extracts clauses and categorizes them by severity (**Normal**, **Warning**, or **Predatory**).
- **🧬 Medical Analysis:** Identifies biomarkers and lab results, comparing them against reference ranges (**Normal**, **High**, or **Low**).
- **📷 Multi-Format OCR:** Supports PNG, JPG, and multi-page PDF files.
- **🎨 Visual Insights:** Color-coded results for quick, intuitive reading.
- **🚀 AI-Driven:** Powered by state-of-the-art open-source models via OpenRouter.

---

## 🛠️ Technology Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **AI Integration:** [OpenRouter API](https://openrouter.ai/) (OpenAI SDK)
- **OCR Engine:** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- **Image Processing:** [OpenCV](https://opencv.org/) & [PIL](https://python-pillow.org/)
- **PDF Conversion:** [pdf2image](https://github.com/Belval/pdf2image) (requires Poppler)

---

## 🚀 Local Installation

### 1. Prerequisites
You must have the following system-level tools installed and added to your **System PATH**:
- **Tesseract OCR:** [Download here](https://github.com/UB-Mannheim/tesseract/wiki)
- **Poppler:** [Download here](https://github.com/oschwartz10612/poppler-windows/releases/)

### 2. Setup
Clone the repository and install the Python dependencies:

```bash
# Clone the repo
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# Install libraries
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## ☁️ Deployment

This app is optimized for deployment on **Streamlit Community Cloud**.

### 📦 System Dependencies (`packages.txt`)
For the cloud server to handle OCR and PDFs, ensure your repository contains a `packages.txt` file with:
```text
tesseract-ocr
poppler-utils
```

### 🔐 Secrets Management
When deploying, add your `OPENROUTER_API_KEY` in the **Advanced Settings > Secrets** section of the Streamlit dashboard in TOML format:
```toml
OPENROUTER_API_KEY = "your_key_here"
```

---

## 📂 Project Structure
```text
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── packages.txt        # System-level dependencies for Linux
├── .gitignore          # Files to exclude from GitHub (like .env)
└── README.md           # Project documentation
```

---

## ⚠️ Disclaimer
This tool is powered by Artificial Intelligence and is intended for **educational and informational purposes only**. It does not provide professional legal or medical advice. Always consult with a qualified lawyer or medical professional before making decisions based on AI-generated analysis.
```

---
