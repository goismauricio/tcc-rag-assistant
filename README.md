
# TCC RAG Assistant 🧠📄

Um assistente de perguntas e respostas sobre documentos PDF usando LLMs com RAG (Retrieval-Augmented Generation).  
Feito como teste com meu TCC, mas funciona com qualquer PDF técnico, artigo ou relatório.

## 🚀 Tecnologias utilizadas

- [Ollama](https://ollama.com/) com LLaMA 3 (ou outro modelo local compatível)
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)

## 💡 O que o projeto faz

Você fornece um arquivo PDF e pode fazer perguntas em linguagem natural sobre o conteúdo.  
O sistema realiza buscas vetoriais (RAG) e responde com base no texto real, citando as fontes (chunks).

## 📦 Como rodar localmente

### Pré-requisitos

- Python 3.10+
- [Ollama instalado localmente](https://ollama.com/)
- Modelos baixados no Ollama (ex: `llama3.2 e nomic-embed-text`)
- Instalar dependências:

```bash
pip install -r requirements.txt
