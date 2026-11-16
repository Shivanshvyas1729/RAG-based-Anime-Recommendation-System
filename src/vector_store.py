import shutil
from pathlib import Path
from typing import List, Optional, Union, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, RETRIEVER_TOP_K
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class VectorStoreBuilder:
    """
    Builds, persists, and loads the Chroma vector store using dense HuggingFace embeddings.
    """
    def __init__(
        self,
        persist_dir: Union[str, Path] = CHROMA_PERSIST_DIR,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        embeddings: Optional[Any] = None
    ):
        self.persist_dir = Path(persist_dir).resolve()
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            logger.info(f"Initializing embeddings with model: {embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def build_and_save_vector_store(
        self,
        documents: List[Document],
        overwrite: bool = True
    ) -> Chroma:
        """
        Creates and persists a Chroma vector database from whole anime documents.
        If overwrite is True, cleanly deletes prior files to prevent duplicate/stale chunks.
        """
        try:
            if overwrite and self.persist_dir.exists():
                logger.warning(f"Overwriting existing vector database at: {self.persist_dir}")
                shutil.rmtree(self.persist_dir)

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Indexing {len(documents)} anime documents into Chroma...")

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_dir)
            )

            total_indexed = vectorstore._collection.count()
            logger.info(f"Chroma vector store built successfully with {total_indexed} records.")
            return vectorstore

        except Exception as e:
            logger.error(f"Failed to build vector store: {str(e)}")
            raise CustomException("Error building and saving vector store", e)

    def load_vector_store(self) -> Chroma:
        """
        Loads an existing Chroma vector store from disk.
        """
        try:
            if not self.persist_dir.exists():
                raise FileNotFoundError(f"Chroma directory {self.persist_dir} does not exist. Please build it first.")

            logger.info(f"Loading Chroma vector database from: {self.persist_dir}")
            vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
            return vectorstore

        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise CustomException("Error loading vector store", e)

    def get_retriever(self, k: int = RETRIEVER_TOP_K):
        """
        Returns a retriever instance configured with top-k nearest neighbors.
        """
        vectorstore = self.load_vector_store()
        return vectorstore.as_retriever(search_kwargs={"k": k})
