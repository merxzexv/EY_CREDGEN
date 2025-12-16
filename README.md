# CREDGEN AI – Intelligent Loan Assistant Chatbot

**CREDGEN AI** is an advanced, conversational banking assistant designed to automate and streamline the loan application process for financial institutions. By orchestrating specialized AI agents (Underwriting, Sales, Fraud, Documentation) with state-of-the-art Large Language Models (LLMs), CREDGEN AI creates a seamless, human-like experience for customers—handling everything from initial inquiry to final sanction letter generation.

---

## 🌟 Key Features

*   **🤖 Multi-Provider AI Engine:**
    *   **Flexible Core:** Switch instantly between **Google Gemini Pro** and **OpenRouter (Gemma 2)** based on your preference or availability.
    *   **Hybrid Intelligence:** Combines the natural language fluency of LLMs with deterministic python logic for critical financial calculations (EMI, interest rates).
*   **💬️ Conversational Application Flow:**
    *   Replaces rigid forms with a friendly chat interface.
    *   Collects personal details, parses uploaded ID documents, and answers FAQs in real-time.
*   **🏦 Bank Personalization (System Tuning):**
    *   **Admin Dashboard:** Banks can "tune" the bot by inputting their specific context (context, policies, current offers) via a simple admin interface.
    *   **Dynamic Injection:** This context is injected into the AI system prompt in real-time, ensuring the bot always speaks with the bank's unique voice and knowledge.
*   **🔒 Automated Fraud Detection:**
    *   Uses **PyOD** (Python Outlier Detection) and machine learning to analyze applicant data and flag high-risk anomalies instantly.
*   **⚡ Instant Underwriting & Scoring:**
    *   Powered by **XGBoost** models trained on financial datasets to generate real-time risk scores and approval decisions.
*   **📄 Automatic Documentation:**
    *   Upon approval, the system auto-generates official **PDF Sanction Letters**, signed and stamped, available for immediate download.

---

## 💬 User Interaction

Interacting with CREDGEN AI is designed to be as simple as chatting with a loan officer.

### Typical Conversation Flow
1.  **Greeting & Intent:**
    *   **User:** "Hi, I need a personal loan."
    *   **Bot:** "Hello! I'd be happy to help with that. Could you please tell me your full name and monthly income?"
2.  **Data Collection & KYC:**
    *   **User:** "My name is John Doe and I earn $5000/month."
    *   **Bot:** "Thanks, John. Please upload your PAN card or ID proof for verification."
    *   (User uploads file -> Bot processes file)
3.  **Underwriting & Offer:**
    *   **Bot:** "Great news! Based on your profile, you are pre-approved. We can offer you a loan of $50,000 at 10.5% interest."
4.  **Closing:**
    *   **User:** "I accept the offer."
    *   **Bot:** "Excellent. I have generated your sanction letter. You can download it below."

---

## ⚙️ Installation & Setup

Follow these steps to get CREDGEN AI running on your local machine.

### 1. Prerequisites
*   **Python 3.9+** installed.
*   **API Keys:** You will need at least one of the following:
    *   [Google AI Studio Key](https://aistudio.google.com/) (for Gemini)
    *   [OpenRouter Key](https://openrouter.ai/) (for OpenRouter models)

### 2. Clone the Repository
```bash
git clone https://github.com/merxzexv/CRED_GEN.git
cd CRED_GEN
```

### 3. Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment
Create a file named `.env` in the root directory and confirm your settings:

```env
# --- LLM Provider Settings ---
LLM_PROVIDER="openrouter"          # Options: "gemini" or "openrouter"
LLM_MODE="enabled"             # Options: "enabled", "disabled", "hybrid"
LLM_SYSTEM_PROMPT="You are CredGen, a professional loan assistant."

# --- API Keys (Fill the corresponding LLM API key or both) ---
GEMINI_API_KEY="your_google_key_here"
OPENROUTER_API_KEY="your_openrouter_key_here"

# --- Security ---
APP_SECRET_KEY="your_random_secret_string"
```

### 6. Run the Application
```bash
python app.py
```
*   **Main Page:** `http://localhost:5000`
*   **Widget Page:** `http://localhost:5000/widget.html`
*   **Admin Dashboard:** `http://localhost:5000/bank/admin/login` (Default User: `admin`, Pass: `admin123`)

---

## 🛠️ Configuration Options

| Variable | Description | Default          |
| :--- | :--- |:-----------------|
| `LLM_PROVIDER` | Choose which AI service to use. | `openrouter`     |
| `LLM_MODE` | `enabled` (AI handles chat), `hybrid` (AI + Rules), or `disabled` (Rules only). | `enabled`        |
| `GEMINI_API_KEY` | Required if using Google Gemini. | -                |
| `OPENROUTER_API_KEY` | Required if using OpenRouter. | -                |
| `APP_SECRET_KEY` | Flask session security key. | `change-this...` |

---

## 🤝 Contributing Guidelines

We welcome contributions!
1.  **Fork** the repository.
2.  Create a **feature branch** (`git checkout -b feature/AmazingFeature`).
3.  **Commit** your changes (`git commit -m 'Add some AmazingFeature'`).
4.  **Push** to the branch (`git push origin feature/AmazingFeature`).
5.  Open a **Pull Request**.

---

## ❓ Troubleshooting & FAQs

**Q: The bot says "AI Service not configured".**
*   **A:** Ensure you have set a valid `GEMINI_API_KEY` or `OPENROUTER_API_KEY` in your `.env` file and that `LLM_PROVIDER` matches the key you provided.

**Q: My sanction letter PDF is blank.**
*   **A:** Ensure the `reportlab` library is installed correctly. Check the console logs for any PDF generation errors.

---

## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software.

---

## 🙏 Acknowledgements & References

*   **Google Gemini** & **OpenRouter** for LLM capabilities.
*   **Flask** framework for the backend.
*   **scikit-learn** & **XGBoost** for the ML models.
*   Inspired by modern fintech automation workflows.
---
**Note**: This is a demonstration system **(made as demo for hackathon: Techathon 6.0 (2025))**.
