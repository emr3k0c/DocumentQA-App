from langchain import PromptTemplate, OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def main():
    llm = OpenAI()
    documents = []
    document_paths = ["/Users/emrekoc/Downloads/doc1.pdf", "/Users/emrekoc/Downloads/doc2.pdf", "/Users/emrekoc/Downloads/doc3.pdf"]
    for pdf in document_paths:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, length_function=len)
    documents = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embeddings)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory,
    )

    while True:
        user_question = input("You: ")  # Get user input

        if user_question.lower() == "stop":
            print("Conversation stopped.")
            break

        response = conversation_chain({'question': user_question})

        # Get the assistant's response

        print("Assistant:", response["answer"])


if __name__ == "__main__":
    main()

