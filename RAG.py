import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
import re
import hashlib
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

DOCS_PATH="./docs/"
DB_PATH = "./db/ChromaDB/"
COLLECTION_NAME = "books_dedup"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
DEVICE = 0
MAX_NEW_TOKENS = 512
K = 5

embedding_model = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=DB_PATH,
    collection_metadata={"hnsw:space": "cosine"}
)

def preprocess_text(text):
    match = re.search(
        r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\n"
        r"(.*?)"
        r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK[^\n]*",
        text,
        flags=re.MULTILINE | re.DOTALL
    )
    if match:
        text = match.group(1)
    text = re.sub(r"(?m)^[ \t]*\*(?:[ \t]*\*)*[ \t]*$", "", text)
    text = re.sub(r"(?m)^[ \t]*\[Illustration\][ \t]*$", "", text)
    text = re.sub(r"\n(?:[ \t]*\n){2,}", "\n\n", text)
    return text.strip()

def load_document(folder_path, db):
    print(f'Loading documents...')

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f'\nFolder {folder_path} does not exist. Create folder and add your favorite books.')

    documents =[]
    hashed = set()
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
             print(f'\nNo documents found in {folder_path}.')
             continue
        if not filename.endswith('.txt'):
             print(f'\n{filename} document type is not supported.')
             continue

        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        stored = db.get(where={'hash_file': file_hash}, include=[])

        if stored['ids']:
            print(f'\n{filename} has been stored in database.')
            continue

        if file_hash in hashed:
            print(f"\n{filename}'s content is detected for duplication.")
            continue
        loader = TextLoader(file_path=file_path, encoding="utf-8")
        documents_load = loader.load()

        for document in documents_load:
                text = document.page_content
                document.metadata['hash_file'] = file_hash
                for key, label in [('title', 'Title'), ('author', 'Author'), ('release_date', 'Release date'), ('language', 'Language')]:
                    match = re.search(rf"(?im)^{label}:[ \t]*(.+)$", text)
                    document.metadata[key] = match.group(1).strip() if match else "Unknown"

                document.metadata['release_date'] = document.metadata['release_date'].split("[")[0].strip()
                document.page_content = preprocess_text(text)

        hashed.add(file_hash)
        documents.extend(documents_load)
    if not documents:
        print("\nNo new documents to process.")
        return []

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}: ")
        print(f"Source: {doc.metadata['source']}")
        print(f'Content length: {len(doc.page_content)}')
        print(f'Content preview: {doc.page_content[:100]}')
        print(f'Metadata: {doc.metadata}')

    return documents

def split_documents(documents):
    print("Splitting documents into chunks.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n','\n',' ','']
    )

    chunks = text_splitter.split_documents(documents)

    positions = {}
    for chunk in chunks:
        source = chunk.metadata['source']
        position = positions.get(source, 0)
        chunk.metadata['chunk_id'] = position
        positions[source] = position+1

    if chunks:

        for i, chunk in enumerate(chunks):
            print(f"\nChunk ID: {chunk.metadata['chunk_id']}")
            print(f"Source: {chunk.metadata['source']}")
            print(f'Length: {len(chunk.page_content)}')
            print(f'Content:')
            print(chunk.page_content)
            print("\n")

        if len(chunks) > 5:
            print(f'\n and {len(chunks) - 5} more chunks')
    return chunks

def menu1():
    documents = load_document(DOCS_PATH, db)

    if documents:
        chunks = split_documents(documents)
        db.add_documents(documents=chunks)
        print(f'{len(documents)} new documents has been stored to database.')

pipeline = HuggingFacePipeline.from_model_id(
    model_id= LLM_MODEL,
    task="text-generation",
    device = DEVICE,
    pipeline_kwargs={
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=pipeline)

def ask_question(question):
    print(f'Question: {question}')
    retriever = db.as_retriever(search_kwargs={"k": K})
    relevant_docs = retriever.invoke(question)

    print(f'\nFound {len(relevant_docs)} documents')

    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    combined_input = f"""Based on the following documents, please answer this question: {question}
    Documents:
    {context}
    Please provide a clear, helpful answer using only the information provided in the documents. If you don't know the answer, just say that you don't know.
    """

    messages = [SystemMessage(content="You are a question-answering assistant for books. Answer the question using only the provided context. Rules: 1. Read all relevant passages before answering. 2. Identify all important reasons related to the question. 3. Combine information from different passages when necessary. 4. Consider the order of events in the story. 5. Keep the answer concise but complete. 6. Do not invent information.")] + [HumanMessage(content=combined_input)]


    if not relevant_docs:
      answer = "I couldn't find relevant information."
      print(f'Answer: {answer}')
      return answer, []
    
    response = model.invoke(messages)
    answer = response.content

    print(f'Answer: {answer}')

    for doc in relevant_docs:
      print(f'\nSource: {doc.metadata["title"]} by {doc.metadata["author"]} –– Chunk {doc.metadata["chunk_id"]}')
      preview = doc.page_content[:250].replace("\n", " ")
      if len(doc.page_content) > 250:
        preview += "..."
      print(f'Evidence: {preview}')

    return answer, relevant_docs

def main():
  print('Library AI Chatbot!')
  while True:
    print("\nMenu:\n1. Load Data\n2. Ask LLM\n3. Exit\n")
    choice = int(input("Enter your choice: "))

    if choice==1:
      menu1()

    elif choice == 2:
      while True:
        question = input('\nYour question: ')

        if question.lower()=='quit' or question=='Quit':
          print('\nGoodbye')
          break

        ask_question(question)
        print('\nType Quit or quit to back to main menu.')
    elif choice == 3:
      print('Thank you for using Library AI Chatbot!')
      break

if __name__ == "__main__":
  main()