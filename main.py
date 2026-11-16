import sys
import argparse
import subprocess
from pathlib import Path

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pipeline.build_pipeline import build_vector_store
from pipeline.pipeline import AnimeRecommendationPipeline
from utils.logger import get_logger

logger = get_logger("Main")

def interactive_chat():
    """Runs an interactive CLI recommendation session."""
    print("=" * 60)
    print("🎬 Welcome to OtakuSensei CLI - AI Anime Recommender")
    print("Type your anime preference or query below. Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    try:
        pipeline = AnimeRecommendationPipeline()
    except Exception as e:
        print(f"\n⚠️  Vector database not ready or failed to load: {e}")
        print("Please build the vector database first using: python main.py --build\n")
        return

    while True:
        try:
            query = input("\n[User] Enter anime preference > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Ja ne! (See you later!) 👋")
                break

            print("\n🤖 Searching database & thinking...")
            result = pipeline.recommend(query)
            print("\n" + "=" * 50)
            print("🎯 RECOMMENDATION:")
            print("=" * 50)
            print(result["recommendation"])
            print("=" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended. Goodbye! 👋")
            break

def main():
    parser = argparse.ArgumentParser(description="OtakuSensei - AI Anime Recommendation System")
    parser.add_argument("--build", action="store_true", help="Build and persist the Chroma vector database")
    parser.add_argument("--query", type=str, help="Run a one-off recommendation query from command line")
    parser.add_argument("--app", action="store_true", help="Launch the Streamlit web application")

    args = parser.parse_args()

    if args.build:
        print("Starting vector store build...")
        build_vector_store()
        print("Vector database built successfully!")
    elif args.app:
        app_path = Path(__file__).parent / "app" / "app.py"
        print(f"Launching Streamlit app from {app_path}...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    elif args.query:
        print(f"Running recommendation for: '{args.query}'\n")
        pipeline = AnimeRecommendationPipeline()
        result = pipeline.recommend(args.query)
        print("=" * 50)
        print("RECOMMENDATION:")
        print("=" * 50)
        print(result["recommendation"])
    else:
        interactive_chat()

if __name__ == "__main__":
    main()
