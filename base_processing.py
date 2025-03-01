import polars as pl
from datetime import datetime
import logging

from datetime import datetime
from pathlib import Path 

class BaseProcessor:
    def __init__(self):
        self.processed_files = {}  # Stores processed data for each file (path -> data)
        self._df = None  # Cached DataFrame

    def _convert_epoch_to_human_readable(self, epoch_time: float) -> str:
        """Convert epoch time (UNIX seconds) to a human-readable format."""
        if epoch_time is None:
            return None
        try:
            return datetime.fromtimestamp(epoch_time).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            logging.warning(f"Invalid epoch_time: {epoch_time}")
            return None

    def _add_human_readable_time(self, file_data: dict) -> dict:
        """Add a human-readable time column if an 'epoch' dataset is present."""
        
        # Find the key that contains 'epoch' (e.g., "/scan/data/epoch")
        epoch_key = next((k for k in file_data if k.endswith("epoch")), None)
        
        if not epoch_key:
            return file_data  # No epoch key found

        epoch_value = file_data[epoch_key]  # Extract the correct key

        if isinstance(epoch_value, dict) and "source" in epoch_value:
            # Keep it lazy: Only resolve when accessed
            file_data["human_readable_time"] = {
                "lazy": lambda: self._convert_epoch_to_human_readable(
                    self.evaluate_lazy_column(file_data, epoch_key)  # Resolve lazily
                )
            }
        else:
            # Directly convert if already loaded
            file_data["human_readable_time"] = self._convert_epoch_to_human_readable(epoch_value)

        return file_data

    
    def _add_filename(self, file_data: dict, file_path: Path) -> dict:
        """Add a 'filename' column to the file_data dictionary."""
        file_data["filename"] = file_path.name
        return file_data
    
    def _ensure_filename_first_column(self):
        """Ensure that the 'filename' column is the first column in the DataFrame."""
        if self._df is not None and "filename" in self._df.columns:
            columns = ["filename"] + [col for col in self._df.columns if col != "filename"]
            self._df = self._df.select(columns)


    def get_dataframe(self, force_reload: bool = False) -> pl.DataFrame:
        """Return the processed data as a Polars DataFrame."""
        self.process_files(force_reload)
        return self._df
    
    

    
    