from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_docs(directory: str):
    loader = DirectoryLoader(directory, glob="**/*.txt")
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_docs("data/kb")
    chunks = split_docs(docs)

    print(f"Loaded documents: {len(docs)}")
    print(f"Generated chunks: {len(chunks)}")
    print()
    print("Sample chunk:")
    print(chunks[0].page_content)