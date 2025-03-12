import polars as pl
from datetime import datetime
import logging
from pathlib import Path 
from typing import Optional, Union
from lazy_dataset import LazyDatasetReference

class BaseProcessor:
    def __init__(self):
        self.processed_files = {}  # Stores processed data for each file (path -> data)


    def _convert_epoch_to_human_readable(self, epoch_dict: dict) -> Optional[str]:
        """Convert epoch time (UNIX seconds) to a human-readable format, resolving lazy references."""

        if not isinstance(epoch_dict, dict) or "lazy" not in epoch_dict:
            logging.warning(f"Expected a dictionary with a 'lazy' key, but got: {epoch_dict}")
            return None

        epoch_reference = epoch_dict["lazy"]

        if not isinstance(epoch_reference, LazyDatasetReference):
            logging.warning(f"Invalid lazy dataset reference: {epoch_reference}")
            return None

        try:
            epoch_times = epoch_reference.load_on_demand()  # Should return a list of floats
            logging.debug(f"Epoch times {epoch_times}")

            if not isinstance(epoch_times, list):  # Ensure it's a list
                logging.warning(f"Expected a list of timestamps, but got: {epoch_times}")
                return None

            # Convert float seconds to integer nanoseconds
            epoch_ns = [int(t * 1_000_000_000) for t in epoch_times]

            return pl.Series("human_readable_time", epoch_ns, dtype=pl.Datetime("ns"))

        except (ValueError, TypeError) as e:
            logging.warning(f"Error converting epoch times {epoch_times}: {e}")
            return None


    def _add_human_readable_time(self, file_data: dict) -> dict:
        """Add a human-readable time column if an 'epoch' dataset is present."""
        
        epoch_key = next((k for k in file_data if k.endswith("epoch")), None)
        if not epoch_key:
            return file_data  # No epoch key found

        epoch_value = file_data[epoch_key]

        # Step 1: Resolve 'source' reference if present
        if isinstance(epoch_value, dict) and "source" in epoch_value:
            source_key = epoch_value["source"]
            if source_key in file_data:
                epoch_value = file_data[source_key]  # Follow the source reference
            else:
                logging.warning(f"Source key {source_key} not found in file_data.")
                return file_data

        # Step 2: Ensure we have a lazy reference
        if isinstance(epoch_value, dict) and "lazy" in epoch_value:
            file_data["human_readable_time"] = self._convert_epoch_to_human_readable(epoch_value)
        else:
            logging.warning(f"Epoch dataset is not a valid LazyDatasetReference: {epoch_value}")

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
            # Try parsing with microseconds
            dt_obj = datetime.strptime(start_time_value, "%Y-%m-%dT%H:%M:%S.%f%z")
        except ValueError:
            try:
                # Fallback: Try parsing without microseconds
                dt_obj = datetime.strptime(start_time_value, "%Y-%m-%dT%H:%M:%S%z")
            except ValueError:
                logging.warning(f"Could not parse start_time: {start_time_value}")
                return file_data  # Return unchanged data if parsing fails

        file_data["human_start_time"] = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        return file_data


    def _add_filename(self, file_data: dict, file_path: Path) -> dict:
        """Add a 'filename' column to the file_data dictionary."""
        file_data["filename"] = file_path.name
        return file_data



