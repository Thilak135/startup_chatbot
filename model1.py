from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import streamlit as st

DB_chroma_PATH = 'vectorstores/db_chroma'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
  """
  Prompt template for QA retrieval for each vectorstore
  """
  prompt = PromptTemplate(template=custom_prompt_template,
                          input_variables=['context', 'question'])
  return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
  qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                         chain_type='stuff',
                                         retriever=db.as_retriever(search_kwargs={'k': 2}),
                                         return_source_documents=True,
                                         chain_type_kwargs={'prompt': prompt}
                                         )
  return qa_chain

#Loading the model
def load_llm():
  # Load the locally downloaded model here
  llm = CTransformers(
      model = r"C:\Users\vidya\OneDrive\Desktop\startup\llama-2-7b-chat.ggmlv3.q8_0.bin",
      model_type="llama",
      max_new_tokens = 512,
      temperature = 0.5
  )
  return llm

#QA Model Function
def qa_bot():
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                     model_kwargs={'device': 'cpu'})
  db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
  llm = load_llm()
  qa_prompt = set_custom_prompt()
  qa = retrieval_qa_chain(llm, qa_prompt, db)

  return qa

#output function
def final_result(query):
  qa_result = qa_bot()
  response = qa_result({'query': query})
  return response

# Streamlit App
def main():
  """ Main function for the Streamlit app """
  st.title("Startup Bot")

  # Text input field for user query
  query = st.text_input("Ask your question:")

  # Run the QA model and display the response
  if query:
    response = final_result(query)
    st.write(response["result"])

if __name__ == "__main__":
  main()
