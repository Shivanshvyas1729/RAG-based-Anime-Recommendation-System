from typing import Dict, Any, Union, List, Optional
from pathlib import Path

from config.config import CHROMA_PERSIST_DIR, RETRIEVER_TOP_K, MODEL_NAME, TEMPERATURE
from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class AnimeRecommendationPipeline:
    """
    High-level orchestration pipeline for anime recommendations.
    Initializes the Chroma retriever and the agentic recommender.
    """
    def __init__(
        self,
        persist_dir: Union[str, Path] = CHROMA_PERSIST_DIR,
        top_k: int = RETRIEVER_TOP_K,
        model_name: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        builder: Optional[VectorStoreBuilder] = None
    ):
        try:
            logger.info("Initializing AnimeRecommendationPipeline...")
            self.builder = builder if builder is not None else VectorStoreBuilder(persist_dir=persist_dir)
            self.retriever = self.builder.get_retriever(k=top_k)
            self.recommender = AnimeRecommender(
                retriever=self.retriever,
                model_name=model_name,
                temperature=temperature
            )
            logger.info("AnimeRecommendationPipeline initialized successfully.")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {str(e)}")
            raise CustomException("Failed to initialize AnimeRecommendationPipeline", e)

    def recommend(self, query: str) -> Dict[str, Any]:
        """
        Processes a user query and returns a dictionary with 'recommendation' and 'retrieved_context'.
        """
        try:
            return self.recommender.recommend(query)
        except Exception as e:
            logger.error(f"Error during recommendation query '{query}': {str(e)}")
            raise CustomException("Recommendation failed", e)

    def recommend_text(self, query: str) -> str:
        """
        Convenience method that returns just the markdown recommendation string.
        """
        result = self.recommend(query)
        return result.get("recommendation", "")

    def get_similar_anime(self, query: str, k: int = 5) -> List[Any]:
        """
        Directly queries the vector store without LLM synthesis.
        Useful for raw context inspection and similarity explorer.
        """
        try:
            retriever = self.builder.get_retriever(k=k)
            return retriever.invoke(query)
        except Exception as e:
            logger.error(f"Error retrieving similar anime for '{query}': {str(e)}")
            raise CustomException("Vector search failed", e)

