from setuptools import setup, find_packages

setup(
    name="anime_recommendation_system",
    version="0.1.0",
    author="Shivansh Vyas",
    description="RAG-based Anime Recommendation System with LangChain, Groq, and Chroma",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-chroma",
        "langchain-groq",
        "chromadb",
        "sentence-transformers",
        "langchain-huggingface",
        "streamlit",
        "pandas",
        "python-dotenv",
        "pillow",
    ],
    python_requires=">=3.10",
)
