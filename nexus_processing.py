import h5py
import polars as pl
from pathlib import Path
import numpy as np
import logging
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any
from base_processing import BaseProcessor
from collections import defaultdict
from dataclasses import dataclass
from lazy_dataset import LazyDatasetReference



# Configure the logger
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)  # Define the logger instance



class NeXusProcessor:
    def __init__(self, file_path: str):
        """Initialise the processor with a NeXus file path."""
        self.file_path = Path(file_path)
        self.directory = self.file_path.parent  # Automatically extract the directory from file_path
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
            
            try:
                if isinstance(obj, h5py.Dataset):
                    #print(f"DEBUG: Found Dataset - {path}")  # Log dataset processing
                    
                    self.structure_dict[path] = {
                        "type": "dataset",
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "unit": obj.attrs.get("unit", ""),
                        "NX_class": obj.attrs.get("NX_class", "")
                    }
                    
                    self._store_dataset(path, obj)

                elif isinstance(obj, h5py.Group):
                    #print(f"DEBUG: Found Group - {path}")  # Log group processing
                    # Record group information
                    self.structure_dict[path] = {
                        "type": "group",
                        "NX_class": obj.attrs.get("NX_class", ""),
                        "children": []
                    }
                    
                    
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

                                if Path(external_file_path).exists():
                                    print(f"DEBUG: External link is valid: {external_file_path}")
                                    try:
                                        with h5py.File(external_file_path, "r") as ext_file:
                                            linked_dataset = ext_file[dataset.path]
                                            self._store_dataset(dataset_path, linked_dataset)
                                    except Exception as e:
                                        logging.warning(f"Skipping broken external link {dataset_path}: {e}")
                                        self.structure_dict[dataset_path] = {"type": "broken_link"}  # Mark as broken link
                                        # Raise an exception for broken external links
                                        raise RuntimeError(f"Broken external link at {dataset_path}: {e}")
                                else:
                                    logging.warning(f"Skipping missing external link: {dataset_path} -> {external_file}")
                                    self.structure_dict[dataset_path] = {"type": "broken_link"}  # Mark as broken link
                                    raise RuntimeError(f"Missing external link: {dataset_path} -> {external_file}")

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
                
                # Record soft link information
                self.structure_dict[path] = {
                    "type": "soft_link",
                    "target": target_path
                }


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
            # Handle string data types (Unicode or byte strings)
            if obj.dtype.kind in {"U", "S"}:  # Unicode or byte strings
                raw_value = obj[()]
                value = raw_value.decode() if isinstance(raw_value, bytes) else raw_value
            
            # Handle scalar datasets (zero-dimensional arrays)
            elif obj.ndim == 0:  # Scalar
                value = obj[()]  # Extract scalar value
            
            # Handle arrays (1D, 2D, 3D, including single-element 1D arrays)
            elif obj.ndim in {1, 2, 3}:  
                file_path = str(self.file_path)  # Store file path
                dataset_name = obj.name  # Store dataset path

                # Mark the dataset as lazy-loaded by storing a LazyDatasetReference
                lazy_reference = LazyDatasetReference(
                    directory=self.directory, 
                    file_name=file_path, 
                    dataset_name=dataset_name
                )
                
                value = {"lazy": lazy_reference}  # Store the reference instead of lambda function
                        
            else:
                logging.warning(f"Skipping dataset {path}: Unsupported shape {obj.shape}")
                return  # Do not store unsupported datasets
            
            # Store value in data_dict
            self.data_dict[path] = {"value": value}

            # Handle unit attribute correctly
            unit = obj.attrs.get("unit", None)
            if unit is not None:
                # Check if the unit is stored as bytes, decode if necessary
                if isinstance(unit, bytes):
                    unit = unit.decode()
                # No decoding needed for string units
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

        #print(75*"\N{aubergine}")
        #print("DEBUG: self.data_dict before processing:")
        #for key, value in self.data_dict.items():
        #    print(f"{key}: {value}")
        #print(75*"\N{aubergine}")

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
        self._df = None  # Cached DataFrame, is eager
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
        self.compare_processed_files()
                
        
        # Store lazy references and soft links properly in `_df`**
        if self.processed_files:
            self._df = self._build_dataframe(resolve=False)  # Keep lazy references
            
    def get_core_metadata(self, force_reload: bool = False) -> pl.LazyFrame:
        """Return a LazyFrame containing only filename, scan_id, scan_command, and human_start_time."""
        
        self.process_files(force_reload)
        if self._df is None:
            raise ValueError("No processed data available.")
        
        return self._df.lazy().select(["filename", "scan_id", "scan_command", "human_start_time"])

    

    def compare_processed_files(self):
        key_types = defaultdict(lambda: defaultdict(set))  # Store encountered types for each key and the files they appeared in
        key_shapes = defaultdict(lambda: defaultdict(set))  # Store encountered shapes for each key (for arrays)
        key_units = defaultdict(lambda: defaultdict(set))  # Store encountered unit attributes for each key
        
        if not self.processed_files:
            print("No files found in processed_files.")
            return

        # Iterate over each file and its associated nested dictionary
        for path, nested_dict in self.processed_files.items():  
            logging.info(f"Checking file: {path}")
            
            if not nested_dict:
                print(f"  Warning: File {path} has an empty dictionary.")
                continue

            # Iterate over the nested dictionary to collect types, shapes, and units for each key
            for key, value in nested_dict.items():
                # Check if the value is a LazyDatasetReference
                if isinstance(value, LazyDatasetReference):
                    # For LazyDatasetReference, defer the loading and handle it later
                    key_types[key][LazyDatasetReference].add(path)
                    # Note: Since the data is lazy-loaded, we can't access its shape directly here.
                    # You could either call load_on_demand() to check the shape, or assume a placeholder.
                    # Assuming that lazy-loading will resolve the shape later
                    key_shapes[key]["lazy-loaded"].add(path)  # Placeholder for lazy-loaded datasets

                # Handle np.ndarray values (they will be lazy-loaded)
                elif isinstance(value, np.ndarray):
                    key_types[key][np.ndarray].add(path)
                    key_shapes[key][value.shape].add(path)  # Track the shape for arrays

                # Handle h5py dataset values
                elif isinstance(value, h5py._hl.dataset.Dataset):
                    try:
                        unit = value.attrs.get("unit")
                        if unit:
                            key_units[key]["unit"].add(unit)
                        key_types[key][type(value)].add(path)
                    except Exception as e:
                        print(f"  Warning: Failed to access 'unit' attribute for key '{key}' in {path}: {e}")
                        key_types[key][type(value)].add(path)

                # Handle other types
                else:
                    key_types[key][type(value)].add(path)

        # Summary of inconsistencies
        logging.info("\nSummary of inconsistencies:")
        found_inconsistencies = False
        for key, types in key_types.items():
            if len(types) > 1:  # More than one type for a key means inconsistency
                found_inconsistencies = True
                print(f"  Inconsistent types for key '{key}':")
                for value_type, paths in types.items():
                    print(f"    Type {value_type} found in files: {', '.join(paths)}")

                    # Check for unit inconsistencies only if units exist for the key
                    if key in key_units and len(key_units[key]) > 1:
                        print(f"    - Units: {', '.join(key_units[key]['unit'])} are inconsistent across files")

                    # Check for shape inconsistencies
                    if key in key_shapes and len(key_shapes[key]) > 1:
                        print(f"    - Shapes: {', '.join(str(shape) for shape in key_shapes[key])} are inconsistent across files")

        if not found_inconsistencies:
            print("  No type inconsistencies found.")


    @staticmethod
    def _resolve_lazy_value(value: Dict[str, Any], key: str) -> Any:
        """Resolves a lazy dataset reference or function if required.
        
        Args:
            value (Dict[str, Any]): Dictionary containing "lazy" or "source" keys.
            key (str): The key corresponding to this value in the dataset.
        
        Returns:
            Any: The resolved dataset, reference, or None if resolution fails.
        """
        lazy_ref = value.get("lazy")
        
        if lazy_ref is None:
            return value.get("source")  # Return source reference if present

        try:
            if isinstance(lazy_ref, LazyDatasetReference):
                return lazy_ref.load_on_demand()  # Resolve LazyDatasetReference
            if callable(lazy_ref):
                return lazy_ref()  # Call function to resolve dataset
            return lazy_ref  # Return as-is if it's neither callable nor a LazyDatasetReference
        except Exception as e:
            logging.warning(f"Failed to resolve lazy dataset for key '{key}': {e}")
            return None  # Return None if resolution fails

    @staticmethod
    def _process_normal_value(value: Any, key: str) -> Any:
        """Validates and processes non-lazy values.

        Args:
            value (Any): The value to process.
            key (str): The key corresponding to this value.

        Returns:
            Any: The validated and processed value.

        Raises:
            ValueError: If an invalid type is encountered.
        """
        if isinstance(value, (float, int, str, type(None), list)):
            return value  # Return valid types directly

        raise ValueError(
            f"Inconsistent type for key '{key}': Expected float, int, str, or None, got {type(value)}"
        )

    def _build_dataframe(self, resolve: bool = False) -> pl.DataFrame:
        """Constructs a Polars DataFrame from processed files.

        Args:
            resolve (bool): Whether to resolve lazy dataset references.

        Returns:
            pl.DataFrame: The constructed Polars DataFrame.
        """
        def process_value(value: Any, key: str) -> Any:
            """Processes each value in the dataset, resolving laziness if needed."""
            
            #logger.debug(f"\N{hot pepper}Key {key}")
            #logger.debug(f"\N{green apple}Value {value}")
            
            
            if isinstance(value, dict):
                return self._resolve_lazy_value(value, key) if resolve else value.get("lazy", value.get("source"))
            return self._process_normal_value(value, key)

        try:
                    
            #TODO: Check what is better later. Both options work here, but one returns a DataFrame, the other imho a LazyFrame.   
            #df=pl.DataFrame([        
            df =  pl.LazyFrame([
                {k: process_value(v, k) for k, v in file_data.items()}
                for file_data in self.processed_files.values()
            ])
            
            #This confirms the result is a realy pl.lazyframe
            logger.debug(10* "\N{red apple}")
            logger.debug(f"{type(df)}")
            logger.debug(f"{df.explain(optimized=True)}")
            logger.debug(10* "\N{red apple}")
            
            
        except ValueError as e:
            logging.error(f"Error building DataFrame: {e}")
            raise  # Re-raise for debugging



    def get_dataframe(self, force_reload: bool = False) -> pl.DataFrame:
        """Return the processed data as a Polars DataFrame with evaluated datasets."""
        self.process_files(force_reload)
        
        # Return cached `_df` if available
        if self._df is not None:
            return self._df

        return self._build_dataframe(resolve=True)  # Now loads actual values



    def get_lazy_dataframe(self, force_reload: bool = False) -> pl.LazyFrame:
        """Return the processed data as a lazy-loaded Polars DataFrame."""

        self.process_files(force_reload)
        
        # Ensure _df is populated without resolving dataset references
        if self._df is None:
            self._df = self._build_dataframe(resolve=False)  # Keep LazyDatasetReference

        return self._df.lazy()  # Convert to LazyFrame for deferred execution


    @staticmethod
    def infer_dtype(df: pl.DataFrame | pl.LazyFrame, col: str):
        """Infer the appropriate Polars dtype based on the first valid dataset reference."""
        
        def resolve_dtype(dataset_ref):
            """Determine dtype from a single dataset reference."""
            if isinstance(dataset_ref, LazyDatasetReference):
                data = dataset_ref.load_on_demand()
                if data is None:
                    return None  # Skip None values
                
                if isinstance(data, list):
                    return pl.List(pl.Float64)  # 1D case
                
                if isinstance(data, np.ndarray):
                    return pl.Array(pl.Float64, data.shape)  # 2D case
                
                
                elif isinstance(dataset_ref, str):
                    return pl.Utf8
                
                elif isinstance(dataset_ref, (int, float)):
                    return pl.Float64 if isinstance(dataset_ref, float) else pl.Int64

            return None  # Default fallback

        if isinstance(df, pl.DataFrame):
            # Iterate over column values and return the first detected dtype (ignoring None)
            for ref in df[col]:
                dtype = resolve_dtype(ref)
                if dtype is not None:  # Skip None values
                    return dtype

            return pl.Object  # Fallback if no valid type was found

        elif isinstance(df, pl.LazyFrame):
            # Collect a small sample of non-null values to determine dtype
            sample_data = (
                df.select(pl.col(col))
                .drop_nulls()
                .limit(10)
                .collect()
                .to_series(0)  # Convert to Polars Series
            )

            # Apply `resolve_dtype` dynamically
            dtypes = [resolve_dtype(value) for value in sample_data if value is not None]
            detected_dtype = dtypes[0] if dtypes else pl.Object  # Use first valid dtype or fallback

            return detected_dtype  # Directly return inferred dtype without redundant map_elements

        else:
            raise TypeError("df must be a Polars DataFrame or LazyFrame")


    def resolve_lazy_references_eagerly(self, df: pl.DataFrame, col_name: str) -> pl.DataFrame:
        """Resolve the LazyDatasetReference objects eagerly in a given column of a DataFrame."""
        
        # Step 1: Infer the correct return dtype for the column
        infer_dtype_val = self.infer_dtype(df, col_name)
        
        print(f"Inferred dtype: { infer_dtype_val}")
        
        # Step 2: Apply transformation to resolve LazyDatasetReferences eagerly
        df_resolved = df.with_columns(
            pl.col(col_name).map_elements(
                lambda ref: ref.load_on_demand() if isinstance(ref, LazyDatasetReference) else None,
                return_dtype= infer_dtype_val
            ).alias(col_name)  # Update the column with resolved data
        )
        
        # Return the DataFrame with the resolved column
        return df_resolved


    
    def resolve_lazy_column(self, df: pl.LazyFrame, dataset_name: str) -> pl.LazyFrame:
        """Evaluate a specific lazy-loaded column in a Polars LazyFrame, 
        resolving LazyDatasetReference objects or other lazy-loaded columns."""
        
        # Ensure required columns exist in the LazyFrame
        schema_names = df.collect_schema().names()
        if 'filename' not in schema_names or dataset_name not in schema_names:
            raise ValueError(f"Column 'filename' or '{dataset_name}' not found in LazyFrame.")


        def resolve_value(value: Any) -> Any:
            """Resolves LazyDatasetReference objects and ensures only expected types are handled."""
            logger.debug(f"Processing value of type {type(value)}: {value}")
                        
            # Pass through valid non-lazy types
            if isinstance(value, (int, float, str, type(None))):
                return value  

            #TODO Polars cannot handles nested objects. If this will here ever a problem, the solution could be to store filename, filepath, dataset name in the pl.df. Metadata as string reference. 
            # Resolve LazyDatasetReference
            if isinstance(value, LazyDatasetReference):
                logger.debug(f"Loading LazyDatasetReference for {value}")
                return value.load_on_demand()
            
            # Handle other potential lazy objects (custom logic for specific types)
            if isinstance(value, pl.LazyList):
                logger.debug(f"Resolving LazyList for {value} in {dataset_name}")
                return value.collect()  # Convert LazyList to a regular list

            # If it's a path, just return the path value (assuming string type)
            if isinstance(value, str) and value.startswith('/'):
                logging.debug(f"Skipping path value in column '{dataset_name}': {value}")
                return value  # Keep path as is
            
            # Unexpected type → Raise error
            raise TypeError(f"Unexpected type in column '{dataset_name}': {type(value)}")



        # Infer the correct return dtype for the column after transformation
        infer_dtype_val = self.infer_dtype(df, dataset_name)
        logger.debug(f'Type in infer_dtype_val of resolve_lazy_column {infer_dtype_val}')
        
        # Apply transformation lazily using map_batches
        df_with_resolved_column = df.with_columns(
            pl.col(dataset_name).map_batches(
                lambda batch: pl.Series([resolve_value(val) for val in batch]),  # Wrap in pl.Series
                return_dtype=infer_dtype_val
            ).alias(dataset_name)
        )


        logger.debug(f"Lazy evaluation set up for column '{dataset_name}'.")
        
        return df_with_resolved_column  # Returns a LazyFrame, though some values may be eagerly evaluated.



    def get_structure_list(self, force_reload: bool = False):
        """Return the hierarchical structure list."""
        self.process_files(force_reload)
        return self.structure_list
    


if __name__ == "__main__":
      
    def test_broken():
        # Initialize the NeXusBatchProcessor with the broken folder path
        damaged_folder = NeXusBatchProcessor("/Users/lotzegud/P08/broken/")
        damaged_folder = NeXusBatchProcessor("/Users/lotzegud/P08/test_folder2/")
        
        
        # Get the DataFrame with regular data (processed files)
        df_damaged = damaged_folder.get_dataframe()
        print("Regular DataFrame (df_damaged):")
        print(type(df_damaged))
        
        
        print(df_damaged.head())
        
        print(30*"\N{pineapple}")
        
        col_name = "/scan/instrument/amptek/data"  # Column where LazyDatasetReference instances are stored
        col_name = '/scan/apd/data'
        col_name='/scan/instrument/collection/exp_t01'
        #col_name='/scan/data/exp_t01'
        col_name='human_readable_time'
        
        df_resolved= damaged_folder.resolve_lazy_references_eagerly(df_damaged, col_name)

        # After applying the transformation, inspect the first row
        print(df_resolved.head())

                
        print(30*"\N{pineapple}")
        
        # Get the DataFrame with lazy-loaded data (evaluated columns)
        df_damaged_lazy = damaged_folder.get_lazy_dataframe()
        print("\N{rainbow}\N{rainbow}\N{rainbow} Lazy-loaded DataFrame (df_damaged_lazy):")
        
        print(3*"\N{hot pepper}", type(df_damaged_lazy) )
        print(df_damaged_lazy.head())
        print(df_damaged_lazy.select(col_name).collect())
        
        print("\N{banana}\N{banana}\N{banana}")
        
        # Now, use the resolve_lazy_column function to resolve the column containing LazyDatasetReference
        df_resolved_lazy = damaged_folder.resolve_lazy_column(df_damaged_lazy, col_name)

        # To see the result of the resolved column
        print(20*"\N{cucumber}")
        print("\N{rainbow}\N{rainbow}\N{rainbow} Resolved DataFrame:")
        print(df_resolved_lazy)
        print('Type of df: ', type(df_resolved_lazy))
        print(20*"\N{cucumber}")
        
        print(df_resolved_lazy.select(col_name).collect())

       
    # Run the test
    test_broken()



    
    def test_raw():
        file_path=Path("/Users/lotzegud/P08/11019623/raw")
        batch_proc=NeXusBatchProcessor(file_path)
        df = batch_proc.get_dataframe()
        
        print(df.head(15))
        
    #test_raw()
    
    def test_healthy():
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
        
        
    #test_healthy()