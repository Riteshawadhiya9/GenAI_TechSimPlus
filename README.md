# GenAI TechSimPlus 🤖

A comprehensive project exploring **Generative AI applications** using LangChain, multiple LLM providers, and interactive interfaces. This project includes AI-powered Q&A bots, advanced prompt engineering, memory management, and various LLM integrations.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks Guide](#notebooks-guide)
- [Technologies Used](#technologies-used)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project serves as a learning and experimentation platform for building intelligent applications with modern Generative AI technologies. It combines:

- **Interactive Web Apps** using Streamlit
- **Educational Notebooks** demonstrating LangChain concepts
- **Multiple LLM Integrations** (Google, Groq, Ollama)
- **Production-ready patterns** for AI applications

---

## ✨ Features

✅ **Q&A Chatbots** - Interactive question-answering systems with streaming responses
✅ **LangChain Integration** - Complete workflow from basic chains to agents
✅ **Memory Management** - Conversation history and context retention
✅ **Structured Output** - JSON parsing and validation from LLMs
✅ **Multiple LLM Providers** - Google, Groq, Ollama support
✅ **Streaming Responses** - Real-time text generation feedback
✅ **AI Agents** - Autonomous agents with tool integration

---

## 📁 Project Structure

```
GenAI_TechTimPlus/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .gitignore                            # Git ignore patterns
│
├── apps/                                 # Production applications
│   ├── 1_QnA_Bot_Streamlit.py           # Streamlit-based Q&A interface
│   └── 1_QnA_ChatBot.py                 # Terminal-based chatbot
│
└── notebooks/                            # Educational Jupyter notebooks
    ├── 1_Basic_LangChain_with_Google.ipynb      # LangChain fundamentals with Google AI
    ├── 2_Prompts_Chain.ipynb                    # Prompt engineering and chaining
    ├── 3_Basic_Memory_With_LangChain.ipynb      # Conversation memory
    ├── 4_Structure_Output.ipynb                 # Structured output parsing
    ├── 5_Ollama_app.ipynb                       # Local LLM with Ollama
    ├── 6_groq_langChain.ipynb                   # Groq LLM integration
    ├── 7_streaming_response.ipynb               # Streaming text generation
    ├── 8_Basic_agents.ipynb                     # AI agents fundamentals
    └── test.ipynb                               # Testing sandbox
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **pip** or **conda**
- **Git**

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/GenAI_TechTimPlus.git
cd GenAI_TechTimPlus
```

### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv env
.\env\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Create a `.env` file in the project root and add your API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**Get API Keys:**
- 🔑 [Google API Key](https://makersuite.google.com/app/apikey)
- 🔑 [Groq API Key](https://console.groq.com)

---

## 💻 Usage

### Run Streamlit Web App

```bash
streamlit run apps/1_QnA_Bot_Streamlit.py
```

This launches an interactive web interface on `http://localhost:8501`

### Run Terminal Chatbot

```bash
python apps/1_QnA_ChatBot.py
```

Interactive Q&A in your terminal

### Run Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and open any notebook to explore concepts

---

## 📚 Notebooks Guide

| Notebook | Description | Level |
|----------|-------------|-------|
| 1_Basic_LangChain_with_Google | LangChain fundamentals + Google AI integration | Beginner |
| 2_Prompts_Chain | Advanced prompt templates and chain composition | Intermediate |
| 3_Basic_Memory_With_LangChain | Conversation memory and context management | Intermediate |
| 4_Structure_Output | Parsing JSON and structured data from LLMs | Intermediate |
| 5_Ollama_app | Running local LLMs with Ollama | Advanced |
| 6_groq_langChain | Groq API integration for faster inference | Intermediate |
| 7_streaming_response | Real-time streaming text generation | Advanced |
| 8_Basic_agents | AI agents with tools and decision-making | Advanced |

**Start with:** `1_Basic_LangChain_with_Google.ipynb` → `2_Prompts_Chain.ipynb` → `3_Basic_Memory_With_LangChain.ipynb`

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **LangChain** | AI/LLM orchestration framework |
| **Streamlit** | Web UI for applications |
| **Jupyter** | Interactive notebook environment |
| **Google Generative AI** | LLM provider (Gemini) |
| **Groq** | Fast LLM inference |
| **Ollama** | Local LLM runtime |
| **Python** | Programming language |

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

```
langchain
streamlit
jupyter
google-generativeai
groq
ollama
python-dotenv
```

To install:
```bash
pip install -r requirements.txt
```

---

## 📝 Key Learnings

This project demonstrates:

1. **LangChain Chains** - Combining LLMs with processing steps
2. **Prompt Engineering** - Crafting effective prompts for better results
3. **Memory Systems** - Maintaining conversation context
4. **Structured Outputs** - Extracting JSON from unstructured LLM responses
5. **Multi-Provider Integration** - Using different LLM APIs
6. **Streaming** - Real-time response generation
7. **Agent Architecture** - Building autonomous AI systems

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Make your changes
4. Commit: `git commit -m "Add YourFeature"`
5. Push: `git push origin feature/YourFeature`
6. Submit a Pull Request

---

## ⚠️ Important Notes

- **API Keys**: Never commit `.env` files or API keys to version control
- **Virtual Environment**: Always use `env/` for dependencies (included in `.gitignore`)
- **Notebook Checkpoints**: `.ipynb_checkpoints` are ignored in `.gitignore`
- **Local Development**: Use the notebooks for experimentation, apps for production

---

## 🐛 Troubleshooting

### Import errors with LangChain?
```bash
pip install --upgrade langchain
```

### Streamlit not found?
```bash
pip install streamlit
```

### API key issues?
- Verify `.env` file is in project root
- Check that environment variables are loaded correctly
- Use `python-dotenv` to load `.env` automatically

---

## 📧 Support

For issues or questions:
- Check existing [GitHub Issues](https://github.com/YOUR_USERNAME/GenAI_TechTimPlus/issues)
- Create a new issue with detailed description
- Include error messages and environment info

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details

---

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [Google Generative AI Docs](https://ai.google.dev)
- [Groq API Docs](https://console.groq.com/docs)
- [Ollama Documentation](https://ollama.ai)

---

**Happy Learning! 🚀**

*Last Updated: May 2026*
