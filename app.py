import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Set up the model and chain
model = OllamaLLM(model="llama3.2")
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit UI
st.set_page_config(page_title="Pizza Restaurant QA", page_icon="🍕")
st.title("🍕 Pizza Restaurant Q&A Bot")
st.write("Ask any question about the pizza restaurant based on customer reviews.")

question = st.text_input("Enter your question:")

if question:
    with st.spinner("Retrieving reviews and generating answer..."):
        reviews = retriever.invoke(question)
        result = chain.invoke({"reviews": reviews, "question": question})
    st.markdown("### 🧾 Answer")
    st.write(result)

    with st.expander("See Retrieved Reviews"):
        st.write(reviews)
