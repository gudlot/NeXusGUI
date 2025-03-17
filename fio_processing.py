import polars as pl
from pathlib import Path
import logging
from base_processing import BaseProcessor
import re

class FioProcessor:
    def __init__(self, file_path: str):
    
        """Initialise the processor with a Fio file path."""
        if not file_path.endswith(".fio"):
            raise ValueError(f"FioProcessor can only handle .fio files: {file_path}")

        self.file_path = file_path
        self.data_dict = {}  # Stores parsed data
    
    def extract_fio_data(self):
        """Parse the ASCII-based Fio file."""
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # Example simple parsing: assuming key-value pairs
        for line in lines:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                self.data_dict[key.strip()] = value.strip()
        
    def to_dict(self):
        """Convert extracted data to a dictionary for Polars."""
        return {"file": self.file_path, **self.data_dict}


class FioBatchProcessor(BaseProcessor):
    def __init__(self, directory: str):
        """Initialise batch processor for a directory of FIO files."""
        super().__init__()
        self.directory = Path(directory)
        self.fio_files = sorted(self.directory.glob("*.fio"))  # Keep a fixed order
        self.processed_files = {}  # Stores processed data for each file (path -> data)
        self._df = None  # Cached DataFrame
        logging.info(f"Found {len(self.fio_files)} FIO files.")

    def update_files(self):
        """Check for new FIO files in the directory and update the file list."""
        all_files = {f for f in self.directory.glob("*.fio")}
        processed_paths = set(self.processed_files.keys())

        # Only retain files that are new
        new_files = all_files - processed_paths

        if new_files:
            logging.info(f"Detected {len(new_files)} new FIO files.")
            self.fio_files = sorted(all_files)  # Replace list instead of extending it

    def process_files(self, force_reload: bool = False):
        """Processes all FIO files, caching results and avoiding redundant work."""
        self.update_files()  # Ensure we have the latest file list

        for file_path in self.fio_files:
            str_path = str(file_path)

            if not force_reload and str_path in self.processed_files:
                continue  # Skip already processed files

            processor = FioProcessor(str_path)
            processor.extract_fio_data()
            file_data = processor.to_dict()

            # Add filename and potential human-readable time
            file_data = self._add_human_readable_time(file_data)
            file_data = self._add_filename(file_data, file_path)

            self.processed_files[str_path] = file_data  # Store processed data

        # Update cached DataFrame
        self._df = pl.DataFrame(list(self.processed_files.values()))
    

    def get_dataframe(self, force_reload: bool = False) -> pl.DataFrame:
        """Return the processed data as a Polars DataFrame."""
        self.process_files(force_reload)
        return self._df

    def get_structure_list(self, force_reload: bool = False):
        """Return a structure overview (if applicable)."""
        self.process_files(force_reload)
        return [{"file": f, "structure": None} for f in self.processed_files.keys()]

    def get_core_metadata(self, force_reload: bool = False) -> pl.DataFrame:
        """Return a DataFrame with filename, scan_id, and scan_command (placeholders for .fio files)."""
        self.process_files(force_reload)
        if self._df is None:
            raise ValueError("No processed data available.")

        pattern = r"_(\d+)\.fio$"
        scan_ids = [re.search(pattern, f.name).group(1) if re.search(pattern, f.name) else "" for f in self.fio_files]

        return pl.DataFrame({
            "filename": [f.name for f in self.fio_files],
            "scan_id": scan_ids,  # Extracted scan_id
            "scan_command": ["" for _ in self.fio_files] , # Empty strings instead of None
            "human_start_time": ["" for _ in self.fio_files] , # Empty strings instead of None
        }).with_columns([
            pl.col("scan_id").cast(pl.String),
            pl.col("scan_command").cast(pl.String)
        ])