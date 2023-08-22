from langchain import PromptTemplate, OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def main():
    user_question = "What is Apple's profit in 2023 q1?"
    documents = []
    document_paths = ["/Users/emrekoc/Downloads/doc1.pdf", "/Users/emrekoc/Downloads/doc2.pdf", "/Users/emrekoc/Downloads/doc3.pdf"]
    for pdf in document_paths:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, length_function=len)
    documents = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embeddings)
    results = db.similarity_search(
        query=user_question,
        n_results=5
    )
    template = """
    You are a chat bot who loves to help people! Given the following context sections, answer the
    question using only the given context. If you are unsure and the answer is not
    explicitly writting in the documentation, say "Sorry, I don't know how to help with that."

    Context sections:
    {context}

    Question:
    {users_question}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "users_question"])
    prompt_text = prompt.format(context=results, users_question=user_question)
    llm = OpenAI()
    print(llm(prompt_text))

    #This part is for different approach to have a conversation history.
    # # memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    # # conversation_chain = ConversationalRetrievalChain.from_llm(
    # #     llm = llm,
    # #     retriever=db.as_retriever(),
    # #     memory = memory
    # # )
    #
    # response = conversation_chain({'question' : user_question})
    # print(response)


if __name__ == "__main__":
    main()

