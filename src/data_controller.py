# NeXusGUI – A GUI for visualising data across multiple NeXus files.
# Copyright (C) 2025 Gudrun Lotze
# This file is part of NeXusGUI.
#
# NeXusGUI is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# NeXusGUI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with NeXusGUI. If not, see <https://www.gnu.org/licenses/>.

__author__ = "Gudrun Lotze"
__copyright__ = "Copyright 2025"
__license__ = "AGPL-3.0"
__status__ = "Development"

import numpy as np
import polars as pl

from collections.abc import Iterable

import logging
from nexus_processing import NeXusBatchProcessor
from fio_processing import FioBatchProcessor
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any
from lazy_dataset import LazyDatasetReference


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BatchProcessorInterface(ABC):
    """Abstract base class for batch processing of data files."""
    
    def __init__(self, file_paths: list[str]):
        self.file_paths = file_paths  # Store file paths

    @abstractmethod
    def get_dataframe(self) -> pl.DataFrame | pl.LazyFrame:
        """Returns a Polars DataFrame (eager) or LazyFrame (lazy)."""
        pass

    @abstractmethod
    def select_column(self, col_name: str) -> pl.Series | pl.LazyFrame:
        """Selects a column from the dataset."""
        pass



class DataController:
    def __init__(
        self, 
        nxs_df: pl.LazyFrame, 
        fio_df: pl.DataFrame, 
        nxs_processor: NeXusBatchProcessor, 
        fio_processor: FioBatchProcessor
    ):
        """Initialise the DataController with metadata and batch processors."""
        self.nxs_df = nxs_df
        self.fio_df = fio_df 
        self.nxs_processor = nxs_processor  # Single Nexus batch processor
        self.fio_processor = fio_processor  # Single Fio batch processor
                
        
    def get_column_names(self, selected_files: list[str]) -> dict[str, str]:
        """
        Returns a dict of unique column names from the selected files, preserving order.

        Args:
            selected_files (list[str]): List of selected filenames.

        Returns:
            dict[str, str]: Ordered unique column names from `.nxs` and `.fio` files.
        """
        column_names = {}

        if not selected_files:
            return column_names

        nxs_selected = any(f.endswith(".nxs") for f in selected_files)
        fio_selected = any(f.endswith(".fio") for f in selected_files)

        logger.debug(f"Processing column names for selected files: {selected_files}")

        if nxs_selected:
            for name in self.nxs_df.collect_schema().names():
                column_names[name] = name

        if fio_selected:
            for name in self.fio_df.columns:
                if name not in column_names:
                    column_names[name] = name

        return column_names
  
    
    def process_selected_files(
        self,
        selected_files: list[str],
        x_column: str,
        y_column: str,
        z_column: str = None,
        normalize: bool = False,
    ) -> pl.DataFrame:
        """
        Processes selected files and returns a DataFrame with the specified columns.
        If normalization is enabled, the y-axis data is divided by the z-axis data.

        Args:
            selected_files (list[str]): List of selected filenames.
            x_column (str): The column name for the x-axis.
            y_column (str): The column name for the y-axis.
            z_column (str): The column name for normalization (optional).
            normalize (bool): Whether to normalize the y-axis data.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the selected columns.
        """
        if not selected_files:
            return pl.DataFrame()  # Return an empty DataFrame if no files are selected

        logger.info(self.nxs_df.limit(5))  # Log the first 5 rows

        # Separate .nxs and .fio files
        nxs_files = [f for f in selected_files if f.endswith(".nxs")]
        fio_files = [f for f in selected_files if f.endswith(".fio")]

        # Initialize empty DataFrames for results
        nxs_data = None
        fio_data = None

        columns = []

        # Always start with 'filename'
        if "filename" not in {x_column, y_column, z_column}:
            columns.append("filename")

        # Collect the rest, preserving order and avoiding duplication
        for col in ("scan_id", x_column, y_column, z_column):
            if col and col not in columns:
                columns.append(col)

            
        # Process .nxs files (lazy)
        if nxs_files:

            # Resolve soft links for all selected columns
        
            logger.debug(f"Before resolving soft links: {columns}")
         
            
            # 1. Get resolved columns
            resolved_columns = self._resolve_soft_links(self.nxs_df, columns)
            logger.debug(f"Column resolution:\nBefore: {columns}\nAfter: {resolved_columns}")

            # 2. Combined collection with optimized sampling
            sample_size = 50
            detection_rows = 5

            combined_sample = (
                self.nxs_df
                .select(resolved_columns)
                .with_row_count()
                .filter(
                    pl.col("row_nr").shuffle().over(resolved_columns[0]) < sample_size
                )
                .drop("row_nr")
                .collect()
            )

            # 3. Enhanced null analysis
            null_stats = combined_sample.select(
                pl.all().null_count().name.prefix("nulls_"),
                pl.all().count().name.prefix("total_")
            )
            print("Column statistics:\n", null_stats)

            # 4. More robust reference detection
            ref_columns = [
                col for col in resolved_columns
                if combined_sample[col].head(detection_rows)
                .is_not_null()
                .any()
                and any(
                    isinstance(val, LazyDatasetReference)
                    for val in combined_sample[col].head(detection_rows)
                    if val is not None
                )
            ]
            logger.info(f"Reference columns found: {ref_columns or 'None'}")

            # 5. Optimized reference loading
            nxs_sample = (
                combined_sample
                if not ref_columns else
                combined_sample.with_columns([
                    pl.col(col).map_elements(
                        lambda x: (
                            x.load_on_demand() 
                            if isinstance(x, LazyDatasetReference) 
                            else x
                        ),
                        return_dtype=pl.Object
                    ).alias(col)
                    for col in ref_columns
                ])
            )
                            
                
            #print("Sample data:\n", nxs_sample)
            ## Show values of first row for all resolved columns
            #try:
            #    first_row = nxs_sample.select(resolved_columns).row(0)
            #    for col, val in zip(resolved_columns, first_row):
            #        print(f"{col}: {val} (type: {type(val)})")
            #except Exception as e:
            #    logger.warning(f"Could not retrieve first row: {e}")
            #
                    
            # Collect metadata
            self.column_metadata = self._analyze_column_metadata(nxs_sample)
            
            # Log metadata summary
            logger.info("Column Metadata Summary:")
            for col, meta in self.column_metadata.items():
                type_info = f"{meta['col_dtype']} (cell: {meta['cell_dtype']})"
                shape_info = f", shape={meta['cell_shape']}" if meta['cell_shape'] else ""
                logger.info(f"- {col:20s}: {type_info}{shape_info}")
                        
                        

        # Process .fio files (eager)
        if fio_files:
            available_columns = [col for col in columns if col in self.fio_df.columns]

            fio_data = (
                self.fio_df
                .select(available_columns)  # Select only available columns
                .sort("scan_id")            # Sort by scan_id
            )

        # Combine results (both eager DataFrames)
        if nxs_data is not None and fio_data is not None:
            # Ensure both frames have the same columns before concatenating
            nxs_data = self.nxs_df.sort("scan_id").select(resolved_columns)
            # Concatenate the nxs_data and fio_data, ensuring both are compatible LazyFrames
            combined_data = pl.concat([nxs_data, fio_data])
        elif nxs_data is not None:
            # Just process nxs_data
            combined_data = self.nxs_df.sort("scan_id").select(resolved_columns)
        elif fio_data is not None:
            # Just process fio_data
            combined_data = fio_data
        else:
            # If neither data is available, return an empty LazyFrame
            combined_data = pl.DataFrame().lazy()

            

        print("DataFrame combined_data Structure:")
        print(combined_data.head())  # Print the first few rows of the DataFrame
        print("\nDataFrame Columns:")
        print(combined_data.collect_schema().names())  # Check the columns
        print("\nDataFrame Types:")
        print(combined_data.collect_schema().dtypes())  # Check data types of the columns

        return combined_data
    
    def _analyze_column_metadata(self, sample_df: pl.DataFrame) -> dict:
        """
        Analyze column metadata using Polars' type system and actual cell contents.

        Returns:
            dict: Nested dictionary with structure {col_name: {metadata_dict}}
        """
        column_metadata = {}
        
        for col in sample_df.columns:
            # Get sample values and first non-null value
            non_null = sample_df[col].drop_nulls()
            sample_values = non_null.head(5).to_list()
            # Ensure sample_values is a valid iterable and non-empty
            if isinstance(sample_values, (list, np.ndarray)) and len(sample_values) > 0:
                first_value = sample_values[0]
            elif isinstance(sample_values, Iterable) and len(sample_values) > 0:
                # Handle iterable types, not just list and ndarray
                first_value = sample_values[0]
            else:
                first_value = None
                
            
            # Get dtype information using Polars' type system
            col_dtype = sample_df.schema[col]
            
            # Initialize base metadata
            col_meta = {
                'col_dtype': str(col_dtype),
                'col_is_numeric': col_dtype.is_numeric(),
                'col_is_integer': col_dtype.is_integer(),
                'col_is_float': col_dtype.is_float(),
                'col_is_temporal': col_dtype.is_temporal(),
                'col_is_string': col_dtype == pl.datatypes.Utf8,  # Checking if the column type is Utf8 (String)
                'col_is_list': isinstance(col_dtype, pl.datatypes.List),  # Checking if column type is List
                'col_is_bool': col_dtype == pl.datatypes.Boolean,  # Checking if the column type is Boolean
                'col_is_null': col_dtype == pl.datatypes.Null,  # Checking if the column type is Null
                'col_is_object': isinstance(col_dtype, pl.datatypes.Object),  # Checking if column type is Object
                'col_is_signed_integer': col_dtype == col_dtype.is_signed_integer(),
                'col_is_unsigned_integer': col_dtype == col_dtype.is_unsigned_integer(),
                'col_is_decimal': col_dtype.is_decimal(),  # Check for Decimal type
                'col_is_nested': col_dtype.is_nested(),    # Check for nested types (Struct)
                'cell_dtype': 'null', 
                'cell_is_numeric': False,
                'cell_is_array': False,
                'cell_shape': None,
                'cell_ndim': 0,
                'null_count': sample_df[col].null_count(),
                'cell_has_size': False,
                'sample_values': sample_values
            }

            # Handle List-type columns
            if isinstance(col_dtype, pl.datatypes.List):
                col_meta['col_is_list'] = True
                col_meta['cell_dtype'] = list
                # Check if the first element in the list is a list and analyze it
                if sample_values:
                    first_list_value = sample_values[0]
                    if isinstance(first_list_value, list):
                        # Optionally, track the shape of the list if needed
                        col_meta['cell_shape'] = (len(first_list_value),)  # Example of list shape
                        col_meta['cell_ndim'] = 1  # Lists are one-dimensional in Polars context
                        col_meta['cell_is_array'] = True
                        # You could also compute statistics on the lists if necessary


            # Handle other column types, including Object and ndarray handling
            elif isinstance(col_dtype, pl.datatypes.Object):
                if isinstance(first_value, np.ndarray):
                    col_meta['cell_has_size'] = True
                    col_meta['cell_dtype'] = np.ndarray
                    col_meta['cell_is_array'] = True
                    col_meta['cell_shape'] = first_value.shape
                    col_meta['cell_ndim'] = first_value.ndim
                   
            else:
                col_meta['unique_count'] = sample_df[col].n_unique()

            # Analyze first value if exists
            if first_value is not None:
                self._analyze_cell_value(first_value, col_meta)

            # Handle string columns
            if col_meta['col_is_string']:
                self._analyze_string_value(sample_values, col_meta)
            
            column_metadata[col] = col_meta

        return column_metadata


    def _analyze_cell_value(self, value: Any, meta: dict) -> None:
        """Analyze individual cell value and update metadata."""
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            meta.update({
                'cell_is_array': True,
                'cell_shape': value.shape,
                'cell_ndim': value.ndim,
                'cell_dtype': str(value.dtype),
                'cell_is_numeric': np.issubdtype(value.dtype, np.number)
            })
        # Handle LazyDatasetReference
        elif isinstance(value, LazyDatasetReference):
            try:
                loaded = value.load_on_demand()
                meta.update({
                    'cell_is_array': True,
                    'cell_shape': getattr(loaded, 'shape', None),
                    'cell_ndim': getattr(loaded, 'ndim', None),
                    'cell_dtype': str(getattr(loaded, 'dtype', 'unknown')),
                    'cell_is_numeric': hasattr(loaded, 'dtype') and np.issubdtype(loaded.dtype, np.number)
                })
            except Exception as e:
                logger.warning(f"Failed to load reference: {e}")
        # Handle Python scalars
        else:
            meta['cell_is_numeric'] = isinstance(value, (int, float, np.number))

    def _analyze_string_value(self, sample_values: list, meta: dict) -> None:
        """Simply check if values are strings and update metadata."""
        if not sample_values:
            return
        
        if isinstance(sample_values[0] , str):
            meta.update({
                'cell_dtype': 'str',  # It's a string
                'cell_is_numeric': False,  # It's not numeric
            })
            

    def _resolve_nested_lazy_datasets(self, lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """Resolve nested LazyDatasets in the given LazyFrame."""
        
        # First, collect the LazyFrame to eagerly resolve any lazy expressions
        eager_frame = lazy_frame.collect()
        
        # Iterate over the columns to handle nested LazyFrames (if any)
        resolved_columns = []
        for c in eager_frame.columns:
            # Check if the column is a LazyFrame, then resolve it by collecting
            if isinstance(eager_frame[c], pl.LazyFrame):
                resolved_columns.append(
                    eager_frame[c].collect().alias(c)
                )
            else:
                resolved_columns.append(
                    pl.col(c).alias(c)  # No transformation needed
                )
        
        # Create a new eager DataFrame with the resolved columns
        return eager_frame.select(resolved_columns)



    def _resolve_soft_links(self, df: pl.LazyFrame, columns: list[str]) -> list[str] :
        """
        Resolves soft links by checking if the column contains only one unique string starting with `/` or None.
        If more than one unique value exists, it raises an error.
        The function returns a list of columns that had their soft links resolved.

        Args:
            df (pl.LazyFrame): The DataFrame containing the columns to check.
            columns (list[str]): List of column names to check for soft links.

        Returns:
            list[str]: List of columns where soft links were resolved.
        """

        print(f"Columns is {columns}")
        
        resolved_columns = []
        
        
        for column in columns:
            # Default: Preserve original column name unless resolved
            result_column = column      
            
            logger.debug(f'\N{hot pepper} \N{hot pepper} \N{hot pepper} Current colum  is {column}')
            print(df.select(column).collect())
            
            # Check if the column exists in DataFrame schema
            if column not in df.schema:
                logger.warning(f"Skipping '{column}' - column not found in LazyFrame schema.")
                resolved_columns.append(result_column)
                continue
                       
            # Only process string (Utf8) columns
            if df.schema[column] == pl.Utf8:
                try:
                    # Check soft-link conditions in one query
                    query = df.select(
                        pl.col(column).str.starts_with("/").any().alias("has_slash"),
                        pl.col(column).drop_nulls().unique().count().alias("unique_count"),
                        pl.col(column).drop_nulls().first().alias("first_value")
                    )
                    has_slash, unique_count, first_value = query.collect().row(0)

                    if has_slash:
                        if unique_count > 1:
                            raise ValueError(f"Ambiguous soft link: '{column}' has multiple unique paths.")
                        if unique_count == 1:
                            result_column = first_value  # Resolve to target path
                            logger.debug(f"Resolved: {column} → {result_column}")
                    else:
                        logger.debug(f"Column '{column}' is a string but not a soft link (no '/' prefix).")
                        # result_column remains unchanged (original name)

                except Exception as e:
                    logger.error(f"Error processing '{column}': {e}")
                    # On error, preserve original name (or re-raise if desired)

            resolved_columns.append(result_column)  # Always append (original or resolved)
        return resolved_columns            

    def _format_and_broadcast_data(
        self,
        data: pl.DataFrame,
        x_column: str,
        y_column: str,
        z_column: str = None,
    ) -> pl.DataFrame:
        """
        Formats and broadcasts data row-wise, ensuring that x_column, y_column, and z_column (if provided)
        have compatible lengths within each row. Metadata (e.g., scan_id, filename) is preserved.

        Args:
            data (pl.DataFrame): The input DataFrame.
            x_column (str): The column name for the x-axis.
            y_column (str): The column name for the y-axis.
            z_column (str): The column name for normalization (optional).

        Returns:
            pl.DataFrame: A DataFrame with properly formatted and broadcasted data.
        """
        # Debugging: Print the structure of the DataFrame
        print(30 * "\N{peacock}")
        print("DataFrame Format_Broadcast Structure:")
        print(data.head())  # Print the first few rows of the DataFrame
        print("\nDataFrame Columns:")
        print(data.columns)  # Check the columns
        print("\nDataFrame Types:")
        print(data.schema)  # Check data types of the columns
        print(30 * "\N{peacock}")



         # Ensure x-axis is numeric if it's epoch time
        if x_column.startswith("/scan/data/epoch"):
            if data[x_column].dtype != pl.List(pl.Float64):
                try:
                    # Convert x_column to a list of floats if not already
                    data = data.with_columns(pl.col(x_column).cast(pl.List(pl.Float64)))
                except pl.ComputeError:
                    raise ValueError(f"Failed to convert {x_column} to a list of floats.")

        # Debugging: Print the first few rows of the x_column
        print("Sample Data in /scan/data/epoch:")
        print(data[x_column].head())
        
        
        print("Before Casting:", data.schema)
        print("Sample Data:", data[x_column].head())

        # Process each row individually
        expanded_data = []
        for row in data.iter_rows(named=True):
            x_values = row[x_column]
            y_values = row[y_column]
            z_values = row[z_column] if z_column else None
            
            # Determine the length of the longest column in this row
            if isinstance(x_values, list):
                length = len(x_values)
            elif isinstance(y_values, list):
                length = len(y_values)
            elif z_column and isinstance(z_values, list):
                length = len(z_values)
            else:
                length = 1
                
            logger.debug(f"x-values : {x_values}")

            # Broadcast x_values, y_values, and z_values to match the length
            x_broadcast = [x_values] * length if not isinstance(x_values, list) else x_values
            y_broadcast = [y_values] * length if not isinstance(y_values, list) else y_values
            z_broadcast = [z_values] * length if z_column and not isinstance(z_values, list) else z_values

            # Add the expanded row(s) to the result
            for j in range(length):
                expanded_data.append({
                    "scan_id": row["scan_id"],
                    "filename": row["filename"],
                    x_column: x_broadcast[j] if isinstance(x_broadcast, list) else x_broadcast,
                    y_column: y_broadcast[j] if isinstance(y_broadcast, list) else y_broadcast,
                    **({z_column: z_broadcast[j]} if z_column else {})
                })

        # Create a new DataFrame with the formatted data
        df = pl.DataFrame(expanded_data)

        # Debugging: Print the structure of the DataFrame
        print(30 * "\N{rainbow}")
        print("DataFrame End Format_Broadcast Structure:")
        print(df.head())  # Print the first few rows of the DataFrame
        print("\nDataFrame Columns:")
        print(df.columns)  # Check the columns
        print("\nDataFrame Types:")
        print(df.schema)  # Check data types of the columns
        print(30 * "\N{rainbow}")

        return df
    
if __name__ == "__main__":
    
    
    def list_files_with_extension(directory: str, extension: str) -> List[str]:
        """
        Returns a list of full file paths for files with the given extension in the specified directory.
        
        :param directory: Path to the directory to search.
        :param extension: File extension to filter by (e.g., '.nxs', '.fio').
        :return: List of full file paths matching the extension.
        """
        if not extension.startswith("."):
            raise ValueError("Extension should start with a dot (e.g., '.nxs', '.fio').")
        
        directory_path = Path(directory)
        
        return [str(directory_path / file.name) for file in directory_path.glob(f"*{extension}")]


    path= "/Users/lotzegud/P08_test_data/healthy2/"
    nxs_processor = NeXusBatchProcessor(path)
    fio_processor = FioBatchProcessor(path)
    
    nxs_metadata_cols = ['filename', 'scan_command', 'scan_id', 'human_start_time']
   
    print(nxs_processor.get_dataframe())
    
    nxs_df_meta = nxs_processor.get_dataframe().select(nxs_metadata_cols).collect()
    print(nxs_df_meta)
    
    nxs_df = nxs_processor.get_dataframe()
    print(nxs_df.select('/scan/data/apd').collect())
    print(nxs_df.select('/scan/apd/data').collect())
    
    fio_df = fio_processor.get_core_metadata()
    print(fio_df)
    
    datacon=DataController(nxs_df, fio_df, nxs_processor, fio_processor)
    
    
    file_selection = list_files_with_extension(path, '.nxs')
    print(file_selection)
    
    x_column= '/scan/instrument/collection/q'
    x_column= '/scan/data/apd'
    y_column = '/scan/instrument/collection/sth_pos'
    z_column= '/scan/ion1/data'
    
    column_names = datacon.get_column_names(file_selection)
    #print(column_names)
    
    datacon.process_selected_files(file_selection, x_column, y_column,  z_column)