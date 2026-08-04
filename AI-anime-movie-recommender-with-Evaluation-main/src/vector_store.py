from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStoreBuilder:
    def __init__(self, csv_path: str = "", persist_dir: str = "chroma_db"):
        self.csv_path = Path(csv_path).resolve() if csv_path else None
        self.persist_dir = Path(persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def build_and_save_vectorstore(self):
        if not self.csv_path or not self.csv_path.exists():
            raise FileNotFoundError(f"CSV path {self.csv_path} does not exist.")

        loader = CSVLoader(
            file_path=str(self.csv_path),
            encoding="utf-8"
        )

        documents = loader.load()

        # Store entire anime documents without arbitrary character splitting
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=str(self.persist_dir)
        )

        return vectorstore

    def load_vector_store(self):
        return Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embedding
        )
