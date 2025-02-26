import h5py
import polars as pl
from pathlib import Path
import numpy as np
import logging
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any
from base_processing import BaseProcessor



class NeXusProcessor:
    def __init__(self, file_path: str):
        """Initialise the processor with a NeXus file path."""
        self.file_path = Path(file_path)
        self.data_dict: Dict[str, Dict[str, Any]] = {}  # Stores dataset paths, values, and units
        self.structure_dict: Dict[str, Any] = {}  # Stores hierarchical structure

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
                    return item, full_path  # Immediately return the first match
                
                # Recursively search deeper
                result = self.find_nxentry(item, full_path, is_root=False)
                if result[0]:  # If a valid NXentry is found in recursion, return it
                    return result

        if is_root:
            print(f"No NXentry found in {self.file_path}.")
        return None, None

    def extract_nexus_data(self):
        """Extracts dataset paths, metadata, and the hierarchical structure."""
        try:
            with h5py.File(self.file_path, 'r') as f:
                nx_entry, nx_entry_path = self.find_nxentry(f)
                if nx_entry is None:
                    print(f"Skipping file {self.file_path}: No NXentry found.")
                    return

                def process_item(name: str, obj):
                    """Processes each object using visititems() and builds structure."""
                    path = f"{nx_entry_path}/{name}"

                    # Build hierarchical structure dictionary
                    levels = path.strip("/").split("/")
                    sub_dict = self.structure_dict
                    for level in levels[:-1]:  
                        sub_dict = sub_dict.setdefault(level, {})

                    # Mark groups and datasets explicitly
                    if isinstance(obj, h5py.Group):
                        sub_dict[levels[-1]] = {"type": "group", "children": {}}
                    else:
                        sub_dict[levels[-1]] = {"type": "dataset", "shape": obj.shape, "dtype": str(obj.dtype)}

                    if isinstance(obj, h5py.Dataset):
                        shape = obj.shape
                        dtype = obj.dtype

                        # Correctly extract string-based datasets
                        if dtype == object or dtype.kind in {"U", "S"}:  # Strings or HDF5 object type
                            value = obj[()]
                            if isinstance(value, bytes):  
                                value = value.decode()  # Convert bytes to string
                        elif shape == () or obj.size == 1:  # Scalar datasets
                            value = obj[()]
                        else:  # Arrays: only store path reference
                            value = f"[Array] {path} (shape={shape}, dtype={dtype})"

                        # Ensure every key in data_dict is a dictionary before adding values
                        self.data_dict.setdefault(path, {})["value"] = value

                        if "unit" in obj.attrs:
                            unit = obj.attrs["unit"]
                            self.data_dict[path]["unit"] = unit  # Only add if it exists
                        else:
                            print(f"Skipping unit for {path}: No unit attribute found")  # Debugging output



                        # Debug specific key
                        if path == "/scan/start_time":
                            print(f"Extracted /scan/start_time: {value} (type={type(value)})")



                # Process all datasets first
                nx_entry.visititems(process_item)

                # Extract scan metadata (scan_command, scan_id) last to avoid duplicates
                scan_metadata = self.extract_scan_metadata(nx_entry_path, f)
                for key, value in scan_metadata.items():
                    if value is not None:
                        self.data_dict[key] = {"value": value}  # Store correctly

                print(f"Data dictionary after scan metadata extraction:\n {self.data_dict}\n")

        except Exception as e:
            logging.warning(f"Error extracting data from {self.file_path}: {e}")


    def extract_scan_metadata(self, nx_entry_path: str, h5file: h5py.File) -> dict:
        """Extracts scan metadata from a NeXus file and returns it as a dictionary."""
        scan_program_name_path = f"{nx_entry_path}/program_name"
        metadata = {}

        try:
            if scan_program_name_path in h5file:
                dataset = h5file[scan_program_name_path]

                for key in ["scan_command", "scan_id"]:
                    try:
                        if key in dataset.attrs:
                            value = dataset.attrs[key]

                            # Handle string-based attributes properly
                            if isinstance(value, bytes):  # If stored as HDF5 bytes
                                value = value.decode()
                            elif isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S"}:
                                value = value[0]  # Extract first element if array of strings

                            metadata[key] = value  # Store in dictionary
                    except Exception as e:
                        print(f"Warning: Failed to extract attribute '{key}' from {scan_program_name_path}: {e}")

        except Exception as e:
            print(f"Error: Failed to access '{scan_program_name_path}' in the NeXus file: {e}")

        return metadata  # Return extracted metadata


    
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
        """Check for new files in the directory and update the file list."""
        all_files = set(self.directory.glob("*.nxs"))
        new_files = all_files - set(self.processed_files.keys())

        if new_files:
            logging.info(f"Detected {len(new_files)} new files.")
            self.nxs_files.extend(new_files)  # Append new files while maintaining order

    def process_files(self, force_reload: bool = False):
        """Processes all NeXus files, caching results and avoiding redundant work."""
        self.update_files()  # Check for new files before processing
        
        for file_path in self.nxs_files:
            str_path = str(file_path)

            # Skip already processed files unless force_reload is enabled
            if not force_reload and str_path in self.processed_files:
                continue

            processor = NeXusProcessor(str_path)
            processor.extract_nexus_data()
            file_data = processor.to_dict()
            
            # Add human-readable time if 'epoch' is present
            file_data = self._add_human_readable_time(file_data)
            
            # Add filename (stripped of path) to the file_data dictionary
            file_data = self._add_filename(file_data, file_path)

            # Cache the processed data
            self.processed_files[str_path] = file_data
            self.structure_list.append({"file": str_path, "structure": processor.structure_dict})

        # Update cached DataFrame after processing
        self._df = pl.DataFrame(list(self.processed_files.values()))
        
        # Ensure 'filename' is the first column
        self._ensure_filename_first_column()

    def get_dataframe(self, force_reload: bool = False) -> pl.DataFrame:
        """Return the processed data as a Polars DataFrame."""
        self.process_files(force_reload)
        return self._df

    def get_lazy_dataframe(self, force_reload: bool = False) -> pl.LazyFrame:
        """Return the processed data as a lazy-loaded Polars DataFrame."""
        self.process_files(force_reload)
        return pl.LazyFrame(self._df)

    def get_structure_list(self, force_reload: bool = False):
        """Return the hierarchical structure list."""
        self.process_files(force_reload)
        return self.structure_list
    


if __name__ == "__main__":
    
    # Initialize the NeXusBatchProcessor with the directory containing .nxs files
    processor = NeXusBatchProcessor("/Users/lotzegud/P08/fio_nxs_and_cmd_tool/")
    
    # Process the files and get the DataFrame
    df = processor.get_dataframe()
    
    # Print the DataFrame
    print("Processed DataFrame:")
    print(df)
    
    # Print the structure list (optional, for debugging)
    #print("\nStructure List:")
    #structure_list = processor.get_structure_list()
    #for item in structure_list:
    #    print(item)
    
    # Print the first few rows of the DataFrame (optional, for debugging)
    #print("\nFirst few rows of the DataFrame:")
    print(df.head())
   
   
    
    print(df["filename"])
    
    print(f'\n')
    print(df.columns)
    print(df.columns[-1])   
    

    
    with h5py.File("/Users/lotzegud/P08/fio_nxs_and_cmd_tool/nai_250mm_02347.nxs", "r") as f:
        if "/scan/start_time" in f:
            dataset = f["/scan/start_time"]
            print("Dataset Shape:", dataset.shape)
            print("Dataset Dtype:", dataset.dtype)
            print("Dataset Value:", dataset[()])
            print("Dataset Attributes:", dict(dataset.attrs))
