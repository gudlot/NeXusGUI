import polars as pl
from datetime import datetime
import logging

from datetime import datetime
from pathlib import Path 
from typing import Optional, Union

class BaseProcessor:
    def __init__(self):
        self.processed_files = {}  # Stores processed data for each file (path -> data)
        self._df = None  # Cached DataFrame

    def _convert_epoch_to_human_readable(self, epoch_time: Union[float, dict]) -> Optional[str]:
        """Convert epoch time (UNIX seconds) to a human-readable format, handling lazy references."""
        
        if epoch_time is None:
            return None

        # Resolve lazy datasets before conversion
        if isinstance(epoch_time, dict) and "lazy" in epoch_time:
            epoch_time = epoch_time["lazy"]()  # Evaluate lazily

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


    def _convert_start_time_to_human_readable(self, file_data: dict) -> dict:
        """Convert start_time (ISO format) to a human-readable format and store it in file_data."""
        for key in file_data:
            if "start_time" in key:  # Check if key contains 'start_time'
                start_time = file_data[key]
                
                if isinstance(start_time, bytes):  # Decode if stored as bytes
                    start_time = start_time.decode()

                try:
                    dt_obj = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%f%z")
                    human_readable = dt_obj.strftime("%Y-%m-%d %H:%M:%S")  # No timezone, otherwise %Z
                    file_data["human_start_time"] = human_readable  # Store in dictionary
                except ValueError:
                    print(f"Warning: Could not parse start_time '{start_time}'")

        return file_data  # Return updated dictionary


    
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
    
    

    
    