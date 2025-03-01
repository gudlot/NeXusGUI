import h5py
import polars as pl
from pathlib import Path
import numpy as np
import logging
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any
from base_processing import BaseProcessor

import traceback

# Configure the logger
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)  # Define the logger instance


class NeXusProcessor:
    def __init__(self, file_path: str):
        """Initialise the processor with a NeXus file path."""
        self.file_path = Path(file_path)
        self.data_dict: Dict[str, Dict[str, Any]] = {}  # Stores dataset paths, values, and units
        self.structure_dict: Dict[str, Any] = {}  # Stores hierarchical structure
        self.nx_entry_path: Optional[str] = None  # Path to NXentry group

    def find_nxentry(self, h5_obj, path: str = "/", is_root: bool = True) -> Tuple[Optional[h5py.Group], Optional[str]]:
        """
        Recursively find the first NXentry group dynamically.
        If no NXentry is found, return (None, None).
        """
        for key in h5_obj.keys():
            full_path = f"{path.rstrip('/')}/{key}"  # Ensure correct path format
            item = h5_obj[key]

            if isinstance(item, h5py.Group):
                if item.attrs.get("NX_class") in [b"NXentry", "NXentry"]:
                    print('\n', item, full_path, '\n')
                    return item, full_path  # Immediately return the first match
                
                # Recursively search deeper
                result = self.find_nxentry(item, full_path, is_root=False)
                if result[0]:  # If a valid NXentry is found in recursion, return it
                    return result
            

        if is_root:
            logging.warning(f"No NXentry found in {self.file_path}.")
            
        return None, None
      
    

    def process(self) -> dict:
        """Extracts data and returns it as a structured dictionary."""
        try:
            with h5py.File(self.file_path, "r", libver="latest") as f:

                nx_entry, self.nx_entry_path = self.find_nxentry(f)
                
                print(100*"\N{rainbow}")
                print(f"nx_entry path: {nx_entry.name}")  # Should be "/scan"
                print(100*"\N{rainbow}")
                
                if nx_entry is None:
                    logging.warning(f"Skipping file {self.file_path}: No NXentry found.")
                    return {}
                
                print("Calling _extract_datasets...")
                self._extract_datasets(nx_entry)
                print("Finished _extract_datasets")
                                
                print(100*"\N{cherries}")
                scan_metadata = self._extract_scan_metadata(nx_entry)
                print(100*"\N{rainbow}")
                for key, value in scan_metadata.items():
                    self.data_dict[key] = {"value": value}
                    
                # Log broken links if any
                broken_links = [key for key, val in self.structure_dict.items() if val.get("type") == "broken_link"]
                if broken_links:
                    logging.warning(f"File {self.file_path} contains {len(broken_links)} broken external links.")


        except Exception as e:
            logging.error(f"Error extracting data from {self.file_path}: {e}")

        return self.to_dict()
    
    
    def _extract_datasets(self, nx_entry: h5py.Group):
        """Extracts datasets and their attributes into data_dict and structure_dict, handling broken external links."""
        
        def process_item(name: str, obj):
            print(30 * "\N{peacock}")  # Confirm execution

            path = f"{self.nx_entry_path}/{name}"
            print(f"DEBUG: Processing {path}")  # Log path being processed

            try:
                if isinstance(obj, h5py.Group):
                    print(f"DEBUG: Found Group - {path}")  # Log group processing

                    nx_class = obj.attrs.get("NX_class", b"").decode() if isinstance(obj.attrs.get("NX_class", ""), bytes) else obj.attrs.get("NX_class", "")
                    print(f"DEBUG: NX_class = {nx_class}")  # Log NX_class

                    if nx_class == "NXdata":
                        signal = obj.attrs.get("signal", None)
                        if isinstance(signal, bytes):
                            signal = signal.decode()

                        print(f"DEBUG: Signal dataset = {signal}")  # Log signal dataset

                        if signal:
                            dataset_path = f"{path}/{signal}"
                            
                            if signal in obj:
                                try:
                                    dataset = obj[signal]
                                    
                                    # 🔹 **Check if it's an external link**
                                    if isinstance(dataset, h5py.ExternalLink):
                                        external_file = dataset.filename
                                        external_file_path = os.path.join(os.path.dirname(self.file_path), external_file)

                                        if os.path.exists(external_file_path):
                                            print(f"DEBUG: External link is valid: {external_file_path}")
                                            # Attempt to open the linked dataset
                                            try:
                                                with h5py.File(external_file_path, "r") as ext_file:
                                                    linked_dataset = ext_file[dataset.path]
                                                    self._store_dataset(dataset_path, linked_dataset)
                                            except Exception as e:
                                                logging.warning(f"Skipping broken external link {dataset_path}: {e}")
                                        else:
                                            logging.warning(f"Skipping missing external link: {dataset_path} -> {external_file}")

                                    elif isinstance(dataset, h5py.Dataset):
                                        self._store_dataset(dataset_path, dataset)

                                except OSError as e:
                                    logging.warning(f"Skipping broken external link: {dataset_path} due to {e}")
                            else:
                                logging.warning(f"Signal dataset '{signal}' not found in {path}")

                elif isinstance(obj, h5py.Dataset):
                    print(f"DEBUG: Found Dataset - {path}")  # Log dataset processing
                    self._store_dataset(path, obj)

            except Exception as e:  # Catch ALL exceptions
                logging.warning(f"Skipping {path} due to {e}")

        nx_entry.visititems(process_item)





    def _store_dataset(self, path: str, obj: h5py.Dataset):
        """Efficiently stores dataset values and unit information in data_dict."""

        shape = obj.shape
        dtype = obj.dtype

        try:
            # Efficient handling of different data types
            if dtype.kind in {"U", "S"}:  # Unicode or byte strings
                value = obj.asstr()[()]
            elif shape == () or obj.size == 1:  # Scalar datasets
                value = obj[()]
            else:
                value = obj  # Stores the reference to the dataset (obj) instead of immediately loading it 

            # Store in data_dict
            self.data_dict[path] = {"value": value}

            # Add unit if available
            unit = obj.attrs.get("unit", None)
            if unit is not None:
                self.data_dict[path]["unit"] = unit

        except OSError as e:
            logging.error(f"Skipping dataset {path} in {self.file_path} due to broken external link: {e}")



    def _extract_scan_metadata(self, nx_entry: h5py.Group) -> dict:
        """Extracts scan metadata from a NeXus file and returns it as a dictionary."""
        
        metadata = {}
        print(100*"\N{cherries}")
    
        
        logging.debug(f"Here we go nx_entry path: {nx_entry.name}")

        # Check if 'program_name' dataset exists
        if "program_name" in nx_entry:
            try:
                program_name_dataset = nx_entry["program_name"]
                metadata["program_name"] = program_name_dataset[()]  # Extract scalar value
                
                # Decode bytes if necessary
                if isinstance(metadata["program_name"], bytes):
                    metadata["program_name"] = metadata["program_name"].decode()

                # Extract attributes using .attrs.get()
                metadata["scan_command"] = program_name_dataset.attrs.get("scan_command", "N/A")
                metadata["scan_id"] = program_name_dataset.attrs.get("scan_id", "N/A")

                logging.debug(f"Extracted metadata: {metadata}")

            except OSError as e:
                logging.warning(f"Skipping 'program_name' due to broken external link: {e}")

        return metadata



    
    def to_dict(self) -> dict:
        """Convert extracted data to a structured dictionary for DataFrame conversion."""
        result = {
            "filename": self.file_path.name,
            "scan_command": self.data_dict.get("scan_command", {}).get("value"),
            "scan_id": self.data_dict.get("scan_id", {}).get("value"),
        }

        # Process all dataset values and units in a single loop
        for key, info in self.data_dict.items():
            value = info.get("value")
            unit = info.get("unit")  # May be None

            result[key] = value
            if unit is not None:  # Only add unit columns if they exist
                result[f"{key}_unit"] = unit

        return result
   

class NeXusBatchProcessor(BaseProcessor):
    def __init__(self, directory: str):
        """Initialise batch processor for a directory of NeXus files."""
        super().__init__()
        self.directory = Path(directory)
        self.nxs_files = list(self.directory.glob("*.nxs"))  # Keep a fixed order
        self.processed_files = {}  # Stores processed data for each file (path -> data)
        self.structure_list = []  # Stores file structure metadata
        self._df = None  # Cached DataFrame
        logging.info(f"Found {len(self.nxs_files)} NeXus files.")

    def update_files(self):
        """Check for new files in the directory and update the file list while maintaining order."""
        all_files = sorted(self.directory.glob("*.nxs"))  # Ensure a sorted order
        existing_files = set(self.processed_files.keys())

        self.nxs_files = [file for file in all_files if str(file) not in existing_files] + [
            file for file in self.nxs_files if str(file) in existing_files
        ]

        if len(self.nxs_files) > len(existing_files):
            logging.info(f"Detected {len(self.nxs_files) - len(existing_files)} new files.")

    def process_files(self, force_reload: bool = False):
        """Processes all NeXus files, caching results and avoiding redundant work."""
        self.update_files()  # Check for new files before processing
        
        # Only clear the structure list if a full reload is requested
        if force_reload:
            self.structure_list.clear()
        
        for file_path in self.nxs_files:
            str_path = str(file_path)

            # Skip already processed files unless force_reload is enabled
            if not force_reload and str_path in self.processed_files:
                continue

            processor = NeXusProcessor(str_path)
            file_data = processor.process()
            
            if not file_data:
                logging.warning(f"Skipping {str_path} due to broken external links or missing data.")
                continue  # Ignore files with broken links
            
            # Add human-readable time if 'epoch' is present
            file_data = self._add_human_readable_time(file_data)
            
            # Add filename (stripped of path) to the file_data dictionary
            file_data = self._add_filename(file_data, file_path)

            # Cache the processed data
            self.processed_files[str_path] = file_data
            self.structure_list.append({"file": str_path, "structure": processor.structure_dict})

        # Update cached DataFrame only if there is processed data
        if self.processed_files:

            # Update cached DataFrame after processing
            self._df = pl.DataFrame(list(self.processed_files.values()))
            
            # Ensure 'filename' is the first column
            self._ensure_filename_first_column()
            
            
    def get_core_metadata(self, force_reload: bool = False) -> pl.LazyFrame:
        """Return a LazyFrame containing only filename, scan_id, and scan_command."""
        #TODO:Remmber u can use latter .collect, i.e. get_core_metadata.collect() to get the values
        
        self.process_files(force_reload)
        if self._df is None:
            raise ValueError("No processed data available.")
        
        return self._df.lazy().select(["filename", "scan_id", "scan_command"])

    def get_dataframe(self, force_reload: bool = False) -> pl.DataFrame:
        """Return the processed data as a Polars DataFrame."""
        self.process_files(force_reload)
        return self._df

    def get_lazy_dataframe(self, force_reload: bool = False) -> pl.LazyFrame:
        """Return the processed data as a lazy-loaded Polars DataFrame."""
        self.process_files(force_reload)
        return self._df.lazy()
    
    def get_structure_list(self, force_reload: bool = False):
        """Return the hierarchical structure list."""
        self.process_files(force_reload)
        return self.structure_list
    


if __name__ == "__main__":
    
    
    # Load NeXusProcessor class (ensure the class definition is included in your script or imported)
    file_path = Path("/Users/lotzegud/P08/11019623/raw/h2o_2024_10_16_01116.nxs")
    
    print("File exists:", file_path.exists())
    print("Absolute path:", file_path.resolve())

    # Create an instance of NeXusProcessor
    processor = NeXusProcessor(file_path)

    # Process the file to extract all data
    data_dict = processor.process()

    # Debug print the extracted scan metadata
    scan_command = data_dict.get("scan_command")
    scan_id = data_dict.get("scan_id")

    logger.debug(f"Final extracted scan metadata: scan_command={scan_command}, scan_id={scan_id}")

    # Print the structured result
    print(data_dict)        