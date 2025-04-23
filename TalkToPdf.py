from ollama import chat
from ollama import ChatResponse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pypdf import PdfReader
import os
import os
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
vector_store = Chroma(
    collection_name="foo",
    embedding_function=HuggingFaceEmbeddings(),
    persist_directory="./chroma_langchain_db"
)
if not os.path.isdir("./chroma_langchain_db"):
    for filename in os.listdir('./documents'):
        print(filename)
        reader = PdfReader('./documents/'+filename)
        pageContext = [page.extract_text() for page in reader.pages]
        docList=[]
        idList=[]
        id=0
        for text in pageContext:
            doc = Document(page_content=text)
            docList.append(doc)
            id=id+1
            idList.append(str(id))
        vector_store.add_documents(documents=docList, ids=idList)

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
while True:
    question = input("Enter your question (type 'quit' to exit): ")
    if question.lower() == 'quit':
        break  # Exit the loop if the user types 'quit'
    else:

        results = vector_store.similarity_search(query=question, k=1)
        context = ""
        for doc in results:
            context += str(doc.page_content)
        response: ChatResponse = chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': context + '\n Based on the context answer the question: ' + question,
            },
        ])
        print(response['message']['content'])

# or access fields directly from the response object
#print(response.message.content)