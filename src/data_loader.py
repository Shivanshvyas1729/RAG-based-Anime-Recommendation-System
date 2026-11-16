from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
from langchain_core.documents import Document

from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class AnimeDataLoader:
    """
    Loads anime catalog data and creates atomic LangChain Document chunks (1 Anime = 1 Document)
    to guarantee that the entire anime context (Title, Overview, Genres, Score) is never fragmented.
    """
    def __init__(self, original_csv: Union[str, Path], processed_csv: Optional[Union[str, Path]] = None):
        self.original_csv = Path(original_csv)
        self.processed_csv = Path(processed_csv) if processed_csv else None

    def _prepare_dataframe(self) -> pd.DataFrame:
        if not self.original_csv.exists():
            raise FileNotFoundError(f"Source anime CSV not found at: {self.original_csv}")

        df = pd.read_csv(self.original_csv, encoding="utf-8", on_bad_lines="skip").dropna(subset=["Name"])
        
        # Handle column spelling variations (sypnopsis vs synopsis)
        synopsis_col = "sypnopsis" if "sypnopsis" in df.columns else "synopsis"
        df[synopsis_col] = df[synopsis_col].fillna("No synopsis available.")
        if "Genres" in df.columns:
            df["Genres"] = df["Genres"].fillna("Unknown")

        required_cols = {"Name", "Genres", synopsis_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Required columns missing in CSV: {missing}")

        # Combine anime fields into a structured semantic block
        df["combined_info"] = (
            "Title: " + df["Name"].astype(str) + "\n" +
            "Overview: " + df[synopsis_col].astype(str) + "\n" +
            "Genres: " + df["Genres"].astype(str)
        )
        return df, synopsis_col

    def load_and_split(self) -> List[Document]:
        """
        Splits data by whole anime entity into atomic LangChain Document objects.
        """
        try:
            logger.info(f"Loading raw anime data from: {self.original_csv}")
            df, _ = self._prepare_dataframe()

            # Optional: export processed CSV if configured
            if self.processed_csv:
                self.processed_csv.parent.mkdir(parents=True, exist_ok=True)
                df[["combined_info"]].to_csv(self.processed_csv, index=False, encoding="utf-8")
                logger.info(f"Saved processed data to: {self.processed_csv}")

            # 1 Anime Entity = 1 Complete Document
            documents = [
                Document(
                    page_content=row["combined_info"],
                    metadata={
                        "title": str(row["Name"]),
                        "genres": str(row.get("Genres", "Unknown")),
                        "score": str(row.get("Score", "N/A")),
                        "mal_id": int(row["MAL_ID"]) if "MAL_ID" in df.columns and pd.notna(row["MAL_ID"]) else None,
                    }
                )
                for _, row in df.iterrows()
            ]

            logger.info(f"Successfully created {len(documents)} complete anime document chunks.")
            return documents

        except Exception as e:
            logger.error(f"Error loading and splitting anime data: {str(e)}")
            raise CustomException("Failed in load_and_split", e)

    def load_and_process(self) -> str:
        """
        Processes the raw CSV, writes the processed CSV file, and returns the path.
        """
        try:
            df, _ = self._prepare_dataframe()
            if not self.processed_csv:
                self.processed_csv = self.original_csv.parent / "anime_updated.csv"

            self.processed_csv.parent.mkdir(parents=True, exist_ok=True)
            df[["combined_info"]].to_csv(self.processed_csv, index=False, encoding="utf-8")
            logger.info(f"Data processed and saved to {self.processed_csv}")
            return str(self.processed_csv)

        except Exception as e:
            logger.error(f"Error in load_and_process: {str(e)}")
            raise CustomException("Failed in load_and_process", e)
