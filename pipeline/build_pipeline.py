from config.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, CHROMA_PERSIST_DIR
from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

def build_vector_store():
    """
    Executes the ingestion and indexing pipeline:
    1. Loads anime data and splits it by entity into atomic documents.
    2. Builds a clean Chroma vector store with HuggingFace embeddings.
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting Anime Recommendation Vector Store Build Pipeline")
        logger.info("=" * 60)

        # Step 1: Data Ingestion & Splitting
        loader = AnimeDataLoader(original_csv=RAW_DATA_PATH, processed_csv=PROCESSED_DATA_PATH)
        documents = loader.load_and_split()
        logger.info(f"Loaded {len(documents)} complete anime documents.")

        # Step 2: Vector Store Building
        builder = VectorStoreBuilder(persist_dir=CHROMA_PERSIST_DIR)
        vectorstore = builder.build_and_save_vector_store(documents=documents, overwrite=True)

        count = vectorstore._collection.count()
        logger.info(f"Pipeline finished! Vector database ready at '{CHROMA_PERSIST_DIR}' with {count} documents.")
        return vectorstore

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise CustomException("Failed to execute build pipeline", e)

if __name__ == "__main__":
    build_vector_store()
