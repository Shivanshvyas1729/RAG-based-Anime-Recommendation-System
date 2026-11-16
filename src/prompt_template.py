from langchain_core.prompts import PromptTemplate

def get_recommender_system_prompt() -> str:
    """
    Returns the system instruction for the LLM agent.
    Explicitly instructs the model to call the anime retriever once, then immediately
    synthesize a structured recommendation without entering recursive tool loops.
    """
    return (
        "You are an expert anime recommendation assistant.\n"
        "Your mission is to help users discover great anime based on their taste, genres, mood, or references.\n\n"
        "Guidelines:\n"
        "1. Always use the 'anime_retriever_tool' to search the database for relevant anime matches.\n"
        "2. Once you receive the tool search results, do NOT make any further tool calls. Immediately synthesize your final recommendations.\n"
        "3. Recommend 2 to 3 best matching anime titles from the retrieved results.\n"
        "4. For each recommended anime, format clearly with:\n"
        "   - **Title** and **Score**\n"
        "   - **Genres**\n"
        "   - **Plot Summary** (concise overview)\n"
        "   - **Why you should watch it / Match Analysis** (how it connects to the user's request)\n"
        "5. If the retrieved results do not match or if the user asks a general anime question, answer politely and helpfully.\n"
        "6. Do not fabricate or invent anime titles that do not exist."
    )

def get_direct_rag_prompt_template() -> PromptTemplate:
    """
    Returns a PromptTemplate for direct (non-tool) RAG generation.
    """
    template = """You are an expert anime recommender.

Based ONLY on the following retrieved anime records, provide a personalized recommendation answering the user's question.

Retrieved Anime Context:
{context}

User's Request:
{question}

Please provide 2-3 structured recommendations with Title, Rating, Genres, a brief summary, and why it fits their request:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])
