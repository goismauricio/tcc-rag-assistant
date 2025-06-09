import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from embedding import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are an academic assistant trained to help with questions related to a final graduation project (TCC) focused on image deblurring. 
Your task is to provide clear, accurate, and technically sound answers about topics such as image processing, blur models, deblurring algorithms, evaluation metrics, datasets, and relevant research papers. 
You may also assist in explaining theoretical concepts, implementation details, and offering suggestions for improvements or future work. 
Always keep your responses formal, concise, and aligned with academic standards. 
When necessary, include equations, citations, or code snippets.
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Função para RAG
def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=10)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3.2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("chunk_id", "Unknown") for doc, _score in results]
    return response_text, sources

# Interface Streamlit
st.set_page_config(page_title="Assistente TCC - Deblurring", page_icon="📘")

st.title("🧠 Assistente de TCC sobre Image Deblurring")
st.write("Faça uma pergunta técnica sobre seu projeto de TCC. O modelo responderá com base nos documentos carregados.")

query = st.text_input("Digite sua pergunta:")

if st.button("Perguntar") and query:
    with st.spinner("Buscando resposta..."):
        resposta, fontes = query_rag(query)

    st.markdown("### 📄 Resposta")
    st.write(resposta)

    st.markdown("### 🔎 Fontes")
    for src in fontes:
        st.markdown(f"- {src}")
