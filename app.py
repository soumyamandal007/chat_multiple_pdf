import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) # for each pdf
        for page in pdf_reader.pages: # for each pages
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size = 1000, chunk_overlap = 200, length_function = len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Process text chunks in batches of 100
    batch_size = 100
    vector_store = None
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        
        if vector_store is None:
            vector_store = FAISS.from_texts(batch, embedding=embeddings)
        else:
            batch_vectors = embeddings.embed_documents(batch)
            vector_store.add_embeddings(
                zip(batch, batch_vectors)
            )
    
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.6)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    st.write("Reply: ", response["output_text"])
    st.session_state.conversation.append({"role": "my_assistant", "content": response["output_text"]})


    

def main():
    load_dotenv()
    #GUI side
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books")
    # st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
        
        
    st.header("Chat with PDFs", divider="blue")
    
    if st.session_state.conversation:
        for message in st.session_state.conversation:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_question = st.chat_input("Ask questions about your documents")
    
    if user_question:
        st.session_state.conversation.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            user_input(user_question)
            
            
    
    print(st.session_state.conversation)

    with st.sidebar: 
        st.subheader(" Your documents")
        pdf_docs = st.file_uploader("Upload your pdfs here ", accept_multiple_files= True)
        if st.button("Process"): # if we press the button
            with st.spinner("Processing "):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                #text chunks
                text_chunks = get_text_chunks(raw_text)
                #the vector store with embeddings
                vectorstore = get_vectorstore(text_chunks)
                st.success("Database created")
                


if __name__ == '__main__':
    main()
