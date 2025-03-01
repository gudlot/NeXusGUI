import h5py
import polars as pl
from pathlib import Path
import numpy as np
import logging
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any
from base_processing import BaseProcessor



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
            with h5py.File(self.file_path, 'r', libver="latest") as f:
                nx_entry, nx_entry_path = self.find_nxentry(f)
                if nx_entry is None:
                    logging.warning(f"Skipping file {self.file_path}: No NXentry found.")
                    return {}
                
                self.nx_entry_path = nx_entry_path  # Store NXentry path

                self._extract_datasets(nx_entry)
                scan_metadata = self._extract_scan_metadata(nx_entry)
                for key, value in scan_metadata.items():
                    self.data_dict[key] = {"value": value}

        except Exception as e:
            logging.error(f"Error extracting data from {self.file_path}: {e}")

        return self.to_dict()
    
    
    def _extract_datasets(self, nx_entry: h5py.Group):
        """Extracts datasets and their attributes into data_dict and structure_dict."""
        def process_item(name: str, obj):
            path = f"{self.nx_entry_path}/{name}"
            #print(f"Processing: {path}, Type: {type(obj)}")

            # Build hierarchical structure dictionary
            levels = path.strip("/").split("/")
            sub_dict = self.structure_dict
            for level in levels[:-1]:
                sub_dict = sub_dict.setdefault(level, {})

            if isinstance(obj, h5py.Group):
                # Store group structure
                sub_dict[levels[-1]] = {"type": "group", "children": {}}

                # Handle NXdata groups explicitly
                nx_class = obj.attrs.get("NX_class", "")
                if isinstance(nx_class, bytes):
                    nx_class = nx_class.decode()
                
                if nx_class == "NXdata":
                    # Extract the 'signal' dataset if defined
                    signal = obj.attrs.get("signal", None)
                    if isinstance(signal, bytes):
                        signal = signal.decode()

                    if signal and signal in obj:
                        dataset = obj[signal]
                        self._store_dataset(f"{path}/{signal}", dataset)

                    # Ensure all datasets inside NXdata are stored
                    for dataset_name in obj.keys():
                        dataset_path = f"{path}/{dataset_name}"
                        dataset = obj[dataset_name]
                        if isinstance(dataset, h5py.Dataset):
                            self._store_dataset(dataset_path, dataset)

            elif isinstance(obj, h5py.Dataset):
                # Store dataset information
                sub_dict[levels[-1]] = {"type": "dataset", "shape": obj.shape, "dtype": str(obj.dtype)}
                self._store_dataset(path, obj)

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

        try:
            if "program_name" in nx_entry:
                obj = nx_entry["program_name"]  # This is a group, not a dataset

                for key in ["scan_command", "scan_id"]:
                    value = obj.attrs.get(key, None)

                    # Decode bytes if necessary
                    if isinstance(value, bytes):
                        value = value.decode()
                    elif isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S"}:
                        value = value[0]  # Extract first element if an array of strings

                    metadata[key] = value  # Store in dictionary
                    
                    # Log extracted values for debugging
                    logger.debug(f"Extracted metadata: {key} = {value}")

        except Exception as e:
            logging.warning(f"Failed to extract scan metadata: {e}")

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
    
    # Initialize the NeXusBatchProcessor with the directory containing .nxs files
    #processor = NeXusBatchProcessor("/Users/lotzegud/P08/fio_nxs_and_cmd_tool/")
    processor = NeXusBatchProcessor("/Users/lotzegud/P08/test_folder/")
    
    # Process the files and get the DataFrame
    #df = processor.get_dataframe()
    # Print the DataFrame
    #print("Processed DataFrame:")
    #print(df.head())
    
    df_lazy= processor.get_lazy_dataframe()
    print(df_lazy.head())
    
    print(100*"\N{hot pepper}")
       
    #print(df_lazy.collect_schema().names)
    for col_name in df_lazy.schema:
        print(col_name)
     
    df= processor.get_dataframe()
    print(df.head())
    
    # Print the structure list (optional, for debugging)
    #print("\nStructure List:")
    #structure_list = processor.get_structure_list()
    #for item in structure_list:
    #    print(item)
    
        
    #print(df["filename"])
    #print(f'\n')
    #print(df.columns)
    #print(df.columns[-1])   
    
    #How to search columns
    #matching_columns = [col for col in df.columns if "scan_" in col]
    #print(matching_columns)
    #processor.update_files()

    
    with h5py.File("/Users/lotzegud/P08/fio_nxs_and_cmd_tool/nai_250mm_02347.nxs", "r") as f:
        if "/scan/start_time" in f:
            dataset = f["/scan/start_time"]
            print("Dataset Shape:", dataset.shape)
            print("Dataset Dtype:", dataset.dtype)
            print("Dataset Value:", dataset[()])
            print("Dataset Attributes:", dict(dataset.attrs))

    print(100*"\N{hot pepper}")

    with h5py.File("/Users/lotzegud/P08/fio_nxs_and_cmd_tool/nai_250mm_02347.nxs", "r") as h5file:
        print(type(h5file["/scan/data"]))
        print(h5file["/scan/data"].name)
        print(h5file["/scan/data"].file.filename)

    with h5py.File("/Users/lotzegud/P08/fio_nxs_and_cmd_tool/nai_250mm_02347.nxs", "r") as f:
        scan_data_group = f["/scan/data"]
        print("Keys in /scan/data:", list(scan_data_group.keys()))  # Check contents
        print("Attributes:", dict(scan_data_group.attrs))  # Check if datasets are stored as attributes



    
    print(100*"\N{pineapple}")

    try:
        with h5py.File("/Users/lotzegud/P08/11019623/raw/h2o_2024_10_16_01116.nxs", "r", libver="latest") as f:
            print("File opened successfully.")
            print("Keys:", list(f.keys()))
    except Exception as e:
        print(f"Error opening file: {e}")


    with h5py.File("/Users/lotzegud/P08/11019623/raw/h2o_2024_10_16_01116.nxs", "r") as f: 
        scan_data_group = f["/scan/data"]
        print("Keys in /scan/data:", list(scan_data_group.keys()))  # Check contents
        print("Attributes:", dict(scan_data_group.attrs))  # Check if datasets are stored as attributes


    print(100*"\N{strawberry}")
    
    
    #sproc=NeXusProcessor("/Users/lotzegud/P08/test_folder/h2o_2024_10_16_01116.nxs")
    #res= sproc._extract_scan_metadata("/scan","h2o_2024_10_16_01116.nxs")
    #res= sproc.process()
    
    
    
    from pathlib import Path
    import h5py

    # Define file path using pathlib
    file_path = Path("/Users/lotzegud/P08/11019623/raw/h2o_2024_10_16_01116.nxs")
     
    '''  
    # Check if the file exists
    if not file_path.exists():
        print(f"Error: File does not exist at {file_path.resolve()}")
    else:
        with h5py.File(file_path, "r") as f:
            def print_structure(name, obj):
                print(f"Path: {name}")
                for attr in obj.attrs:
                    print(f"  - Attribute: {attr} = {obj.attrs[attr]}")

            f.visititems(print_structure)
            
    
            
    with h5py.File(file_path, "r") as f:
        for key in f:
            try:
                obj = f[key]
            except OSError as e:
                print(f"Broken link found: {key} - {e}")
            
    with h5py.File(file_path, "r") as f:
        def check_links(name, obj):
            if isinstance(obj, h5py.SoftLink):
                print(f"Soft link found: {name} -> {obj.path}")

        f.visititems(check_links)
        
    '''
    if not file_path.exists():
        print(f"Error: File does not exist at {file_path.resolve()}")
    else:
        with h5py.File(file_path, "r") as f:
            def search_for_paths(name, obj):
                # Check attributes
                for attr, value in obj.attrs.items():
                    if isinstance(value, (bytes, str)) and "nxs" in value.lower():
                        print(f"Potential file reference in attribute: {name}/{attr} -> {value}")

                # Check dataset contents
                if isinstance(obj, h5py.Dataset) and obj.dtype.kind in {'S', 'U'}:  # String types
                    data = obj[()]
                    if isinstance(data, np.ndarray):  # Handle array of strings
                        for item in data:
                            if isinstance(item, (bytes, str)) and "nxs" in item.lower():
                                print(f"Potential file reference in dataset: {name} -> {item}")
                    elif isinstance(data, (bytes, str)) and "nxs" in data.lower():
                        print(f"Potential file reference in dataset: {name} -> {data}")

            f.visititems(search_for_paths)

    print(100*"\N{banana}")
    print(100*"\N{banana}")
    
    
    # h5ls -r /Users/lotzegud/P08/11019623/raw/h2o_2024_10_16_01116.nxs
    # that allows you to inspect the structure of an HDF5 file without using Python.
    # -r recursive mode
    
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