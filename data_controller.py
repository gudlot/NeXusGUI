import numpy as np
import polars as pl
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataController:
    def __init__(self, nxs_df, fio_df):
        self.nxs_df = nxs_df
        self.fio_df = fio_df
        
        
    def get_column_names(self, selected_files: list[str]) -> dict[str, None]:
        """
        Returns a dictionary of column names for the selected files.
        """
        column_names = {}

        if selected_files:
            nxs_files = [f for f in selected_files if f.endswith(".nxs")]
            fio_files = [f for f in selected_files if f.endswith(".fio")]

            # Extract column names from nxs_df (lazy)
            if nxs_files:
                column_names |= {col: None for col in self.nxs_df.schema.keys()}
                #logger.debug(f"Column names from nxs_df: {list(column_names.keys())}")

            # Extract column names from fio_df (eager)
            if fio_files:
                column_names |= {col: None for col in self.fio_df.columns}
                #logger.debug(f"Column names from fio_df: {list(column_names.keys())}")


        return column_names
    
    
    def process_selected_files(self, selected_files: list[str], x_column: str, y_column: str, z_column: str = None) -> pl.DataFrame:
        """
        Processes selected files and returns a DataFrame with the specified columns.

        Args:
            selected_files (list[str]): List of selected filenames.
            x_column (str): The column name for the x-axis.
            y_column (str): The column name for the y-axis.
            z_column (str): The column name for normalization (optional).

        Returns:
            pl.DataFrame: A Polars DataFrame containing the selected columns.
        """
        if not selected_files:
            return pl.DataFrame()  # Return an empty DataFrame if no files are selected

        # Separate .nxs and .fio files
        nxs_files = [f for f in selected_files if f.endswith(".nxs")]
        fio_files = [f for f in selected_files if f.endswith(".fio")]

        # Initialize empty DataFrames for results
        nxs_data = None
        fio_data = None

        # Columns to select
        columns = ["scan_id", x_column, y_column]
        if z_column:
            columns.append(z_column)

        # Process .nxs files (lazy)
        if nxs_files:
            nxs_data = (
                self.nxs_df
                .filter(pl.col("filename").is_in(nxs_files))  # Filter selected files
                .select(columns)  # Select required columns
                .sort("scan_id")  # Sort by scan_id
            )

        # Process .fio files (eager)
        if fio_files:
            fio_data = (
                self.fio_df
                .filter(pl.col("filename").is_in(fio_files))  # Filter selected files
                .select(columns)  # Select required columns
                .sort("scan_id")  # Sort by scan_id
            )

        # Combine results (lazy + eager)
        if nxs_data is not None and fio_data is not None:
            combined_data = pl.concat([nxs_data.collect(), fio_data])  # Collect nxs_data (lazy → eager)
        elif nxs_data is not None:
            combined_data = nxs_data.collect()  # Collect nxs_data (lazy → eager)
        elif fio_data is not None:
            combined_data = fio_data  # Already eager
        else:
            return pl.DataFrame()  # Return an empty DataFrame if no data is found
        
        print("DataFrame combined_data Structure:")
        print(combined_data.head())  # Print the first few rows of the DataFrame
        print("\nDataFrame Columns:")
        print(combined_data.columns)  # Check the columns
        print("\nDataFrame Types:")
        print(combined_data.dtypes)  # Check data types of the columns
    
        

        return combined_data

    def _format_and_broadcast_data(self, data: pl.DataFrame, x_column: str, y_column: str) -> pl.DataFrame:
        """
        Formats and broadcasts data, resolving NeXus soft links only when necessary.

        Args:
            data (pl.DataFrame): The input DataFrame.
            x_column (str): The column name for the x-axis.
            y_column (str): The column name for the y-axis.

        Returns:
            pl.DataFrame: A DataFrame with properly formatted and broadcasted data.
        """
        # Resolve soft links at the last moment
        x_column = self.resolve_soft_links(x_column)
        y_column = self.resolve_soft_links(y_column)

        # Convert Polars DataFrame to Pandas for easier manipulation
        data_pd = data.to_pandas()
        
        print("DataFrame combined_data Structure:")
        print(data_pd.head())  # Print the first few rows of the DataFrame
        print("\nDataFrame Columns:")
        print(data_pd.columns)  # Check the columns
        print("\nDataFrame Types:")
        print(data_pd.dtypes)  # Check data types of the columns
    
        

        # Ensure x-axis is numeric if it's epoch time
        if x_column.startswith("/scan/data/epoch"):
            try:
                x_data = data_pd[x_column].astype(float)  # Convert to numerical timestamps
            except ValueError:
                raise ValueError(f"Failed to convert {x_column} to numeric epoch time.")
        else:
            x_data = self._expand_column(data_pd[x_column])

        # Map filenames to categorical numerical values
        if y_column == "filename":
            unique_filenames = {name: idx for idx, name in enumerate(sorted(data_pd[y_column].unique()))}
            y_data = data_pd[y_column].map(unique_filenames)  # Replace filenames with numeric category
        else:
            y_data = self._expand_column(data_pd[y_column])

        # Ensure x and y data have the same length
        if len(x_data) != len(y_data):
            raise ValueError(f"Length mismatch: x-axis ({len(x_data)}) and y-axis ({len(y_data)}) must have the same length.")

        # Create a new DataFrame with the formatted data
        return pl.DataFrame({x_column: x_data, y_column: y_data})


    def resolve_soft_links(self, column_name: str) -> str:
        """Resolve NeXus soft links within `nxs_df` if applicable."""
        schema = self.nxs_df.schema  # Get the schema of the LazyFrame
        
        if column_name in schema and isinstance(column_name, str) and column_name.startswith("/"):
            logger.debug(f"Resolving soft link: {column_name}")
            return column_name  # Replace this with actual resolution logic if needed
        
        return column_name



    def _expand_column(self, column_data) -> list:
        """
        Expands and formats a column of data to ensure compatibility for plotting.

        Args:
            column_data: The column data (can be strings, ints, floats, 1D arrays, etc.).

        Returns:
            list: A list of expanded and formatted values.
        """
        expanded_values = []

        for value in column_data:
            if isinstance(value, str):
                # Convert strings to categorical values (e.g., "string1" → 0, "string2" → 1)
                unique_strings = {s: i for i, s in enumerate(set(column_data))}
                expanded_values.append(unique_strings[value])
            elif isinstance(value, np.ndarray):
                if value.shape == (1,):
                    # Broadcast scalar arrays
                    expanded_values.append(value.item())
                elif len(value) == len(column_data):
                    # Use the array as-is
                    expanded_values.extend(value.tolist())
                else:
                    st.error(f"Cannot process array: Invalid shape {value.shape}.")
                    return []
            elif isinstance(value, (int, float)):
                # Broadcast scalar values
                expanded_values.append(value)
            else:
                st.error(f"Cannot process value: Unsupported data format {type(value)}.")
                return []

        return expanded_values