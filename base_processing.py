import polars as pl
from datetime import datetime
import logging
from pathlib import Path 
from typing import Optional, Union
from lazy_dataset import LazyDatasetReference

class BaseProcessor:
    def __init__(self):
        self.processed_files = {}  # Stores processed data for each file (path -> data)
        self._df = None  # Cached DataFrame

    def _convert_epoch_to_human_readable(self, epoch_time: Union[float, str, bytes, LazyDatasetReference]) -> Optional[str]:
        """Convert epoch time (UNIX seconds) to a human-readable format, resolving lazy references."""
        
        if epoch_time is None:
            return None

        # Resolve lazy dataset reference
        if isinstance(epoch_time, LazyDatasetReference):
            epoch_time = epoch_time.load_on_demand()

        # Handle string/byte conversions
        if isinstance(epoch_time, (str, bytes)):
            try:
                epoch_time = float(epoch_time)  
            except (ValueError, TypeError):
                logging.warning(f"Invalid epoch_time: {epoch_time}")
                return None

        if isinstance(epoch_time, (float, int)):
            try:
                return datetime.fromtimestamp(epoch_time).strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError) as e:
                logging.warning(f"Error converting epoch_time: {epoch_time} | Error: {e}")
                return None
        
        logging.warning(f"Invalid epoch_time format: {epoch_time}")
        return None

    def _add_human_readable_time(self, file_data: dict) -> dict:
        """Add a human-readable time column if an 'epoch' dataset is present."""
        
        epoch_key = next((k for k in file_data if k.endswith("epoch")), None)
        if not epoch_key:
            return file_data  

        epoch_value = file_data[epoch_key]

        if isinstance(epoch_value, dict) and "source" in epoch_value:
            source_key = epoch_value["source"]
            if source_key in file_data:
                epoch_value = file_data[source_key]  # Resolve source reference
        
        file_data["human_readable_time"] = self._convert_epoch_to_human_readable(epoch_value)
        return file_data

    def _convert_start_time_to_human_readable(self, file_data: dict) -> dict:
        """Convert start_time (ISO format) to a human-readable format and store it in file_data."""
        
        start_time_key = "/scan/start_time"
        if start_time_key not in file_data:
            return file_data  

        start_time_value = file_data[start_time_key]

        if isinstance(start_time_value, bytes):
            start_time_value = start_time_value.decode()

        try:
            dt_obj = datetime.strptime(start_time_value, "%Y-%m-%dT%H:%M:%S.%f%z")
            file_data["human_start_time"] = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            logging.warning(f"Warning: Could not parse start_time '{start_time_value}'")

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
