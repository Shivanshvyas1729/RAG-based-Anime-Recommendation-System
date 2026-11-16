from typing import Dict, Any, List
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from config.config import MODEL_NAME, MODEL_PROVIDER, TEMPERATURE, MAX_TOKENS
from src.prompt_template import get_recommender_system_prompt
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

def build_anime_retriever_tool(retriever):
    """
    Constructs the LangChain tool wrapping the Chroma anime retriever.
    """
    @tool
    def anime_retriever_tool(query: str) -> str:
        """
        Searches the anime knowledge base for titles matching user preferences, genres, themes, or plot queries.
        Returns complete anime descriptions with Title, Score, Genres, and Overview.
        """
        logger.info(f"anime_retriever_tool invoked with query: '{query}'")
        docs = retriever.invoke(query)
        if not docs:
            logger.info("No anime records retrieved.")
            return "No relevant anime found in the database."

        formatted_items: List[str] = []
        for d in docs:
            title = d.metadata.get("title", "Unknown Title")
            score = d.metadata.get("score", "N/A")
            genres = d.metadata.get("genres", "N/A")
            formatted_items.append(
                f"Title: {title}\n"
                f"Score: {score}\n"
                f"Genres: {genres}\n"
                f"{d.page_content}"
            )
        return "\n\n---\n\n".join(formatted_items)

    return anime_retriever_tool


class AnimeRecommender:
    """
    Orchestrates the tool-calling recommendation flow using Groq-hosted LLMs.
    """
    def __init__(
        self,
        retriever,
        model_name: str = MODEL_NAME,
        model_provider: str = MODEL_PROVIDER,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ):
        logger.info(f"Initializing AnimeRecommender with model: {model_name} via {model_provider}")
        self.retriever = retriever
        self.llm = init_chat_model(
            model_name,
            model_provider=model_provider,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.retriever_tool = build_anime_retriever_tool(self.retriever)
        self.model_with_tools = self.llm.bind_tools([self.retriever_tool])
        self.system_prompt = get_recommender_system_prompt()

    def recommend(self, query: str) -> Dict[str, Any]:
        """
        Executes the agentic recommendation cycle:
        1. Query analysis & tool decision.
        2. Vector search execution.
        3. Recommendation synthesis with fallback resolution.
        """
        try:
            logger.info(f"Processing recommendation request: '{query}'")
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query)
            ]

            # Step 1: Model inspects query
            ai_msg = self.model_with_tools.invoke(messages)
            messages.append(ai_msg)

            retrieved_context = ""

            # Step 2: Handle tool calls if requested
            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")

                    tool_result = self.retriever_tool.invoke(tool_args)
                    retrieved_context += tool_result + "\n\n"

                    messages.append(
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"]
                        )
                    )

                # Step 3: Synthesize final answer with retrieved evidence
                final_response = self.model_with_tools.invoke(messages)
                
                # Robust extraction handling models with reasoning_content
                answer = final_response.content or final_response.additional_kwargs.get("reasoning_content", "")
                if not answer and final_response.tool_calls:
                    # Fallback to direct prompt if model attempted repeated tool calls
                    logger.warning("Model attempted nested tool call; falling back to direct synthesis.")
                    fallback_prompt = (
                        f"Based ONLY on this retrieved anime data:\n{retrieved_context}\n\n"
                        f"Answer the user request: {query}"
                    )
                    answer = self.llm.invoke(fallback_prompt).content

                if isinstance(answer, list):
                    answer = "\n".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in answer])

                logger.info("Recommendation generated successfully.")
                return {
                    "query": query,
                    "recommendation": answer.strip(),
                    "retrieved_context": retrieved_context.strip()
                }

            # Direct response if no tool call was required
            logger.info("Model answered without tool call.")
            return {
                "query": query,
                "recommendation": ai_msg.content.strip(),
                "retrieved_context": "No database search needed."
            }

        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            raise CustomException("Recommendation execution failed", e)
