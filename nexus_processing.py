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
                    #print('\n', item, full_path, '\n')
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
                
                if nx_entry is None:
                    logging.warning(f"No NXentry found in {self.file_path}. Skipping.")
                    return {}
                
                
                self._extract_datasets(nx_entry)
                               
                                
            
                scan_metadata = self._extract_scan_metadata(nx_entry)
           
                self.data_dict.update(scan_metadata)
               
                    
                # Log broken links
                broken_links = [k for k, v in self.structure_dict.items() if v.get("type") == "broken_link"]
                if broken_links:
                    logging.warning(f"File {self.file_path} contains {len(broken_links)} broken external links.")


        except Exception as e:
            logging.error(f"Error extracting data from {self.file_path}: {e}")

        return self.to_dict()
    
    
    


    def _extract_datasets(self, nx_entry: h5py.Group):
        """Extracts datasets, preserving hierarchy and soft links, and handling broken external links."""

        def process_item(name: str, obj):
            path = f"{self.nx_entry_path}/{name}"
            #print(f"DEBUG: Processing {path}")  # Log path being processed

            try:
                if isinstance(obj, h5py.Dataset):
                    #print(f"DEBUG: Found Dataset - {path}")  # Log dataset processing
                    self._store_dataset(path, obj)

                elif isinstance(obj, h5py.Group):
                    #print(f"DEBUG: Found Group - {path}")  # Log group processing
                    nx_class = obj.attrs.get("NX_class", b"").decode() if isinstance(obj.attrs.get("NX_class", ""), bytes) else obj.attrs.get("NX_class", "")

                    if nx_class == "NXdata":
                        signal = obj.attrs.get("signal", None)
                        if isinstance(signal, bytes):
                            signal = signal.decode()

                        #print(f"DEBUG: Signal dataset = {signal}")  # Log signal dataset

                        if signal and signal in obj:
                            dataset_path = f"{path}/{signal}"
                            dataset = obj[signal]

                            if isinstance(dataset, h5py.ExternalLink):
                                # Handle external link dataset
                                external_file = dataset.filename
                                external_file_path = os.path.join(os.path.dirname(self.file_path), external_file)

                                if os.path.exists(external_file_path):
                                    print(f"DEBUG: External link is valid: {external_file_path}")
                                    try:
                                        with h5py.File(external_file_path, "r") as ext_file:
                                            linked_dataset = ext_file[dataset.path]
                                            self._store_dataset(dataset_path, linked_dataset)
                                    except Exception as e:
                                        logging.warning(f"Skipping broken external link {dataset_path}: {e}")
                                        self.structure_dict[dataset_path] = {"type": "broken_link"}  # Mark as broken link
                                else:
                                    logging.warning(f"Skipping missing external link: {dataset_path} -> {external_file}")
                                    self.structure_dict[dataset_path] = {"type": "broken_link"}  # Mark as broken link

                            elif isinstance(dataset, h5py.Dataset):
                                self._store_dataset(dataset_path, dataset)
                            else:
                                logging.warning(f"Signal dataset '{signal}' is neither a dataset nor an external link.")

            except Exception as e:
                logging.warning(f"Skipping {path} due to {e}")

        #Visit all regular datasets/groups first**
        nx_entry.visititems(process_item)

        #Visit and process all soft links separately**
        def process_link(name: str, link_obj):
            """Process soft links and store them with a unified lazy format."""
            path = f"{self.nx_entry_path}/{name}"

            if isinstance(link_obj, h5py.SoftLink):
                target_path = link_obj.path
                #print(f"DEBUG: Found soft link {path} -> {target_path}")

                # Store soft link reference in a unified format
                #self.data_dict[path] = {
                #    "lazy": partial(self._resolve_lazy_dataset, target_path)  # Standardised format
                #}
                # Store the reference to the target dataset
                self.data_dict[path] = {
                    "source": target_path  # Keeps the reference path without causing lazy evaluation issues
                }

        # Use visititems_links to process **only links**, including soft links
        nx_entry.visititems_links(process_link)



    def _store_dataset(self, path: str, obj: h5py.Dataset):
        """Efficiently stores dataset values and unit information in data_dict."""
        try:
                # Determine how to handle the dataset
            if obj.dtype.kind in {"U", "S"}:  # Unicode or byte strings
                raw_value = obj[()]
                value = raw_value.decode() if isinstance(raw_value, bytes) else raw_value
            elif obj.shape == () or obj.size == 1:  # Scalar datasets
                value = obj[()]
            elif obj.ndim == 1 or obj.ndim == 2 or obj.ndim == 3:  # Lazy-load 1D & 2D arrays
                file_path = str(self.file_path)  # Store file path
                dataset_name = obj.name  # Store dataset path

                def load_on_demand():
                    """Reopen file and load dataset when needed."""
                    with h5py.File(file_path, "r") as f:
                        return pl.Series(dataset_name, f[dataset_name][:])

                value = {"lazy": load_on_demand}  # Mark as lazy function

            else:
                logging.warning(f"Skipping dataset {path}: Unsupported shape {obj.shape}")
                return  # Do not store unsupported datasets
            
            # Store in data_dict
            self.data_dict[path] = {"value": value}

            # Handle unit attribute correctly
            unit = obj.attrs.get("unit", None)
            if isinstance(unit, bytes):  # Ensure decoding if unit is stored as bytes
                unit = unit.decode()
            if unit is not None:
                self.data_dict[path]["unit"] = unit

        except OSError as e:
            logging.error(f"Skipping dataset {path} in {self.file_path} due to broken external link: {e}")



    def _extract_scan_metadata(self, nx_entry: h5py.Group) -> dict:
        """Extracts scan metadata from a NeXus file and returns it as a dictionary."""
        
        metadata = {}
        program_name_path = f"{self.nx_entry_path}/program_name"
               
        #logging.debug(f"Here we go nx_entry path: {nx_entry.name}")

        # Check if 'program_name' dataset exists
        if "program_name" in nx_entry:
            try:
                program_name_dataset = nx_entry["program_name"]
                metadata[program_name_path] = {
                    "value" : 
                    (program_name_dataset[()].decode() if isinstance(program_name_dataset[()], bytes)
                    else program_name_dataset[()]
                )
                }

                # Extract attributes efficiently and ensure they are stored with "value"
                metadata.update({
                    key: {"value": (value.decode() if isinstance(value, bytes) else str(value))}
                    for key in ["scan_command", "scan_id"]
                    if (value := program_name_dataset.attrs.get(key, "N/A")) is not None
                })

                #logging.debug(f"Extracted metadata: {metadata}")

            except OSError as e:
                logging.warning(f"Skipping 'program_name' due to broken external link: {e}")

        return metadata


    
    def to_dict(self) -> dict:
        """Convert extracted data to a structured dictionary for DataFrame conversion."""
        
        # Extract essential metadata
        result = {
            "filename": self.file_path.name,
            "scan_command": self.data_dict.get("scan_command", {"value": "N/A"}).get("value", "N/A"),
            "scan_id": self.data_dict.get("scan_id", {"value": "N/A"}).get("value", "N/A"),
        }

        print(75*"\N{aubergine}")
        print("DEBUG: self.data_dict before processing:")
        for key, value in self.data_dict.items():
            print(f"{key}: {value}")
        print(75*"\N{aubergine}")

        # Iterate over extracted datasets and metadata
        for key, info in self.data_dict.items():
            if key in {"scan_command", "scan_id"}:  # Prevent overwriting
                continue  

            if isinstance(info, dict):
                if "lazy" in info:
                    result[key] = {"lazy": info["lazy"]}  # Preserve lazy references
                elif "value" in info:
                    #result[key] = info["value"]  # Store immediate values
                    raw_value = info["value"]
                    result[key] = raw_value.decode() if isinstance(raw_value, bytes) else raw_value  # 🔹 Decode bytes
                elif "source" in info:  
                    result[key] = {"source": info["source"]}  # Keep soft link reference as source path
            else:
                result[key] = info  # Directly store other values (e.g., scalars, strings)

            # Store unit if available
            if isinstance(info, dict) and "unit" in info:
                result[f"{key}_unit"] = info["unit"]

        #print("DEBUG: Final result:", result)  
        print(20*"\N{blueberries}")
        return result
    
    def _resolve_lazy_dataset(self, path: str):
        """Resolves a dataset or group path when accessed lazily, with debugging prints."""
        try:
            with h5py.File(self.file_path, "r") as f:
                obj = f[path]
                
                print(f"DEBUG: Resolving {path}")  # Track path resolution

                if isinstance(obj, h5py.Dataset):
                    print(f"DEBUG: {path} is a dataset. Loading contents...")
                    return obj[:]  # Load dataset contents

                elif isinstance(obj, h5py.Group):
                    print(f"DEBUG: {path} is a group. Listing contents:")
                    
                    # Print each entry inside the group
                    for name, item in obj.items():
                        print(f"    - {name}: {type(item)}")

                    # Load all datasets inside the group into a dictionary
                    return {
                        name: dset[:] for name, dset in obj.items() if isinstance(dset, h5py.Dataset)
                    }

                else:
                    print(f"DEBUG: {path} is an unknown type: {type(obj)}")
                    return None  

        except Exception as e:
            print(f"ERROR: Failed to resolve {path}: {e}")
            return None


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
        
        logging.info(f"Current path in update_files: {self.directory}")
        
        all_files = sorted(self.directory.glob("*.nxs"))  # Retrieve and sort all files
        existing_files = set(self.processed_files.keys())

        self.nxs_files = [file for file in all_files if str(file) not in existing_files] + [
            file for file in self.nxs_files if str(file) in existing_files
        ]

        if len(self.nxs_files) > len(existing_files):
            logging.info(f"Detected {len(self.nxs_files) - len(existing_files)} new files.")




    def process_files(self, force_reload: bool = False):
        """Processes all NeXus files, caching results and avoiding redundant work."""
        
        logging.info(f"Current path in process_files: {self.directory}")
        self.update_files()  # Check for new files before processing
        logging.info(f"After update files path in process files: {self.directory}")
        
        if force_reload:
            self.structure_list.clear()
            self.processed_files.clear()
        
        for file_path in self.nxs_files:
            str_path = str(file_path)

            if not force_reload and str_path in self.processed_files:
                continue  # Skip if already processed

            processor = NeXusProcessor(str_path)
            file_data = processor.process()


            if not file_data:
                logging.warning(f"Skipping {str_path} due to broken external links or missing data.")
                continue  
            
            # Add human-readable time if 'epoch' is present
            file_data = self._add_human_readable_time(file_data)
            
            # Add human-readable start_time if present
            file_data = self._convert_start_time_to_human_readable(file_data)

            # Store data **without forcing eager evaluation**
            self.processed_files[str_path] = file_data  
            self.structure_list.append({"file": str_path, "structure": processor.structure_dict})

        logging.info(f"Processed {len(self.processed_files)} NeXus files.")
        
        # Store lazy references and soft links properly in `_df`**
        if self.processed_files:
            self._df = self._build_dataframe(resolve=False)  # Keep lazy references
            
    def get_core_metadata(self, force_reload: bool = False) -> pl.LazyFrame:
        """Return a LazyFrame containing only filename, scan_id, scan_command, and human_start_time."""
        
        self.process_files(force_reload)
        if self._df is None:
            raise ValueError("No processed data available.")
        
        return self._df.lazy().select(["filename", "scan_id", "scan_command", "human_start_time"])


    def _resolve_lazy_value(self, value):
        """Helper function to resolve lazy datasets and handle soft links.

        - If `value` is a dict containing `"lazy"`, it evaluates the dataset.
        - If `value` is a dict containing `"source"`, it keeps the reference.
        - Otherwise, returns the value as-is.
        """
        if isinstance(value, dict):
            if "lazy" in value:
                try:
                    return value["lazy"]()  # Evaluate dataset
                except Exception as e:
                    logging.warning(f"Failed to resolve lazy dataset: {e}")
                    return None  # Return None to avoid crashes
            if "source" in value:
                return f"Reference: {value['source']}"  # Keep soft link as reference

        return value  # Return direct values


    def _build_dataframe(self, resolve: bool = False) -> pl.DataFrame:
        """Constructs a Polars DataFrame from processed files.
        
        - If `resolve` is True, it evaluates lazy-loaded datasets.
        - If `resolve` is False, it keeps lazy references.
        """
        def resolve_value(value):
            """Ensure values are either resolved or stored as references."""
            if isinstance(value, dict):
                if "lazy" in value and resolve:
                    return value["lazy"]()  # Evaluate dataset
                elif "lazy" in value:
                    return "<Lazy Dataset>"  # Placeholder for lazy references
                if "source" in value:
                    return value["source"]  # Keep soft link as a reference
            return value  # Return normal values

        return pl.DataFrame([
            {k: resolve_value(v) for k, v in file_data.items()}
            for file_data in self.processed_files.values()
        ])


    def get_dataframe(self, force_reload: bool = False) -> pl.DataFrame:
        """Return the processed data as a Polars DataFrame with evaluated datasets."""
        self.process_files(force_reload)
        
        # Return cached `_df` if available
        if self._df is not None:
            return self._df

        # Use `_resolve_lazy_value` for evaluation
        return self._build_dataframe(resolve=True)  # Now loads actual values



    def get_lazy_dataframe(self, force_reload: bool = False) -> pl.LazyFrame:
        """Return the processed data as a lazy-loaded Polars DataFrame."""
        self.process_files(force_reload)
        
        if self._df is None:
            raise ValueError("No processed data available.")
        
        return self._df.lazy()

    
    def evaluate_lazy_column(self, df: pl.DataFrame, column_name: str) -> pl.Series:
        """Evaluate a specific lazy-loaded column in a Polars DataFrame."""
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")

        return df[column_name].map_elements(self._resolve_lazy_value, return_dtype=pl.Object)
    

    def get_structure_list(self, force_reload: bool = False):
        """Return the hierarchical structure list."""
        self.process_files(force_reload)
        return self.structure_list
    


if __name__ == "__main__":
      
    def test_broken():
    
        broken=NeXusProcessor("/Users/lotzegud/P08/broken/h2o_2024_10_16_01116.nxs")
        broken.process()
        
        broken_data = NeXusBatchProcessor("/Users/lotzegud/P08/broken/")
        df_broken= broken_data.get_dataframe()
        
        print(df_broken.head())    
        
    test_broken()
        
    
    def test_health():
        file_path = Path("/Users/lotzegud/P08/fio_nxs_and_cmd_tool/nai_250mm_02349.nxs")
        print(f"File exists: {file_path.exists()}")
        print(f"Absolute path: {file_path.resolve()}")
            
        sproc=NeXusProcessor(file_path)
        sproc.process()
        res= sproc.to_dict()

        
        print(75*"\N{mango}")
        for key, value in res.items(): 
            print(key, value)
            if key == "/scan/ion_bl/mode":
                print('This is ', type(value))
            if key == "/scan/start_time":
                print('This is start_time ', type(value))
        print(75*"\N{mango}")

        '''
        #h5ls -r /Users/lotzegud/P08/healthy/nai_250mm_02290.nxs
        
        # Check if the file exists
        if not file_path.exists():
            print(f"Error: File does not exist at {file_path.resolve()}")
        else:
            with h5py.File(file_path, "r") as f, open("/Users/lotzegud/P08/hdf5_structure.txt", "w") as output_file:
                def print_structure(name, obj):
                    output_file.write(f"Path: {name}\n")
                    for attr in obj.attrs:
                        output_file.write(f"  - Attribute: {attr} = {obj.attrs[attr]}\n")

                f.visititems(print_structure)

        print("Output saved to hdf5_structure.txt")
            
        '''

        # Initialize the NeXusBatchProcessor with the directory containing .nxs files
        processor = NeXusBatchProcessor("/Users/lotzegud/P08/fio_nxs_and_cmd_tool/")
        #processor = NeXusBatchProcessor("/Users/lotzegud/P08/healthy/")
        
        # Process the files and get the DataFrame
        df = processor.get_dataframe()
                
        #for col in df.columns:
        #    print(col)   
        
        
        # Print the DataFrame
        print("Processed DataFrame:")
        
        pl.Config.set_tbl_rows(10)  # Set maximum displayed row
        #pl.Config.set_tbl_rows(None)  # Show all rows
        print(df)
        
        
        print(75*"\N{hot pepper}")


    
        
        print(75*"\N{blueberries} ")
        
        df_lazy= processor.get_lazy_dataframe()
        print(df_lazy.head())
        
        #print(df_lazy.collect_schema().names)
        #for col_name in df_lazy.schema:
        #   print(col_name)
        #df= processor.get_dataframe()
        #print(df.head())

        #print(75*"\N{mango}")
        ##now eager, and it looks like ti works 
        #df_eager = df_lazy.collect()
        #print(df_eager.head())  # Now it prints real data
        
        
        hdf5_path = df["/scan/apd/data"][0]  # Get the path from the first row of the column

        print("HDF5 path:", hdf5_path)  # Print the HDF5 path