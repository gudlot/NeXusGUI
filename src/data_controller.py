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
import logging
from nexus_processing import NeXusBatchProcessor
from fio_processing import FioBatchProcessor
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
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

        columns = ["scan_id", x_column, y_column]
        if z_column:
            columns.append(z_column)

        # If any of x_column, y_column, or z_column (if provided) are not "filename", add "filename" to the list
        if "filename" not in {x_column, y_column} and (not z_column or z_column != "filename"):
            columns.insert(0, "filename")  # Ensures "filename" appears first

            
        # Process .nxs files (lazy)
        if nxs_files:

            # Resolve soft links for all selected columns
            
            resolved_columns = self._resolve_soft_links(self.nxs_df, columns)
            logger.debug(f"Before resolving soft links: {columns}")
            logger.debug(f"After resolving soft links: {resolved_columns}")
            
            # Step 1: Get null counts (as you're already doing)
            col_stats = nxs_df.select([
                pl.col(col).null_count().alias(f"{col}_nulls") 
                for col in resolved_columns
            ]).collect()

            print("Null counts:\n", col_stats)
            
            # Step 2: Sample non-null rows for reference detection
            detection_sample = (
                nxs_df
                .select(resolved_columns)
                .filter(
                    pl.any_horizontal(pl.col(resolved_columns).is_not_null())
                )
                .head(5)
                .collect()
            )

            # Step 3: Identify reference columns
            ref_columns = [
                col for col in resolved_columns
                if any(
                    isinstance(val, LazyDatasetReference)
                    for val in detection_sample[col]
                    if val is not None
                )
            ]
            print(f"Columns requiring load_on_demand: {ref_columns}")
            
            #nxs_df (Lazy) 
            #→ [select] → LazyFrame1 
            #→ [filter] → LazyFrame2 
            #→ [map_batches] → LazyFrame3 
            #→ [collect] → nxs_sample (DataFrame)
            
            #Original sampling with added reference loading
            nxs_sample = (
                nxs_df
                .select(resolved_columns)
                .filter(pl.arange(0, pl.len()).shuffle().over(resolved_columns[0]) < 50)
                .map_batches(lambda batch: pl.DataFrame({
                    col: (
                        batch[col].map_elements(
                            lambda x: x.load_on_demand(),
                            return_dtype=pl.Object  # or a more specific dtype if known
                        ) if col in ref_columns 
                        else batch[col]
                    )
                    for col in batch.columns
                }))
                .collect()
            )
            print("Sample data:\n", nxs_sample)
            value = nxs_sample.select(pl.col('/scan/apd/data')).row(0)[0]
            print(value)
            print(type(value))
            

         
         
            
            logger.debug(10*"\N{strawberry}")
            logger.debug(resolved_nxs_df)

            logger.debug(10*"\N{strawberry}")

           

        # Process .fio files (eager)
        if fio_files:
            # No need to resolve soft links for .fio files
            fio_data = (
                self.fio_df
                .filter(pl.col("filename").is_in(fio_files))  # Filter selected files
                .select(columns)  # Select required columns
                .sort("scan_id")  # Sort by scan_id
            )

        # Combine results (both eager DataFrames)
        if nxs_data is not None and fio_data is not None:
            combined_data = pl.concat([nxs_data, fio_data])
        elif nxs_data is not None:
            combined_data = nxs_data
        elif fio_data is not None:
            combined_data = fio_data
        else:
            ombined_data = pl.DataFrame()  # Return an empty DataFrame if no data is found

            

        print("DataFrame combined_data Structure:")
        print(combined_data.head())  # Print the first few rows of the DataFrame
        print("\nDataFrame Columns:")
        print(combined_data.columns)  # Check the columns
        print("\nDataFrame Types:")
        print(combined_data.dtypes)  # Check data types of the columns

        return combined_data
                

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