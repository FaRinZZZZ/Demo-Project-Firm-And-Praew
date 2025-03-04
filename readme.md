# D-DEV Assistant

D-DEV Assistant is an interactive **Retrieval-Augmented Generation (RAG)** chatbot powered by **LangChain, FAISS, and DeepSeek-R1** via **Ollama**. It retrieves knowledge from a **local text file (`data/data.txt`)**, processes it using **vector embeddings**, and answers questions in a **tsundere-style persona** (D-DEV).

---

## 🚀 Features

- **Conversational UI:** Uses Streamlit's `st.chat_message()` for a chat-like experience.
- **Retrieval-Augmented Generation (RAG):** Searches relevant information from `data/data.txt` before responding.
- **Customizable Personality:** D-DEV is a **tsundere AI assistant**, responding with playful sarcasm while still being helpful.
- **Interactive Feedback:** Shows a loading spinner (`st.spinner`) while generating responses.

---

## 📦 Installation

### 1️⃣ Install Dependencies

Ensure you have Python 3.8+ and **pip** installed, then run:

```bash
pip install streamlit langchain langchain_community langchain-ollama faiss-cpu
```

_If using a GPU-based FAISS install, replace `faiss-cpu` with `faiss-gpu`._

### 2️⃣ Install Ollama (If Not Installed)

D-DEV Assistant uses **Ollama** to run the `deepseek-r1` LLM. If you haven't installed **Ollama**, install it from:

[https://ollama.com/](https://ollama.com/)

Then pull the DeepSeek-R1 model:

```bash
ollama pull deepseek-r1
```

### 3️⃣ Clone the Repository (Optional)

If this project is hosted on GitHub, clone it:

```bash
git clone https://github.com/yourusername/d-dev-assistant.git
cd d-dev-assistant
```

---

## 🏃‍♂️ Running the App

To start D-DEV Assistant, run:

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501** to start chatting with **D-DEV**.

---

## 📂 Project Structure

```
📦 d-dev-assistant/
 ├── 📂 data/
 │   ├── data.txt  # Knowledge base (retrieved by FAISS)
 ├── app.py        # Streamlit chatbot UI
 ├── README.md     # Documentation (this file)
 ├── requirements.txt # Dependencies (optional)
```

---

## ⚙️ Configuration

### 🔹 Changing the Knowledge Base

- The assistant retrieves information from **data/data.txt**.
- To update the knowledge base, **edit or replace `data.txt`** and restart the app.

### 🔹 Adjusting Chunking Settings

- The text is **split into chunks** for better retrieval.
- Modify chunk size & overlap in `app.py`:
  ```python
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
  )
  ```

### 🔹 Changing the Personality (System Prompt)

To modify **D-DEV** personality, edit this section:

```python
system_message = SystemMessagePromptTemplate.from_template(
    "You are D-DEV, a super cute and **tsundere** assistant robot. "
    "You're sassy, playful, a bit cold at first, but secretly you *care*. "
    "Include tsundere phrases ('Baka', 'You idiot'), and short helpful answers. "
)
```

---

## 🚀 Future Enhancements

- **Memory Storage:** Implement conversational memory for multi-turn conversations.
- **Streaming Output:** Enable token-by-token streaming for more dynamic responses.
- **Multi-file Support:** Allow PDFs & multiple text sources for RAG.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit PRs.

---

## 📝 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## 🏆 Credits

Developed by **[FaRinZZZZ](https://github.com/farinzzzz)** and **[PRAEWA CHOOBANNA](https://github.com/praewery)**, powered by **LangChain, FAISS, and Ollama**.
