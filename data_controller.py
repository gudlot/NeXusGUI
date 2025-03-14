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

        logger.info(self.nxs_df.limit(5).collect())  # Log the first 5 rows

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
        # Process .nxs files (lazy)
        if nxs_files:
            # Identify columns to resolve
            columns_to_resolve = [col for col in (x_column, y_column, z_column) if col]

            # Resolve soft links for all selected columns
            resolved_nxs_df = self._resolve_soft_links(self.nxs_df, columns_to_resolve)

            # Now filter and select the required columns
            nxs_data = (
                resolved_nxs_df
                .filter(pl.col("filename").is_in(nxs_files))  # Filter selected files
                .select(columns)  # Select required columns
                .sort("scan_id")  # Sort by scan_id
            )

            print(30 * "\N{pineapple}")
            logger.debug(f"nxs_data type: {type(nxs_data)}")
            nxs_data_eager = nxs_data.collect()
            print("DataFrame nxs_data_eager Structure:")
            print(nxs_data_eager.head())  # Print the first few rows of the DataFrame
            
                        # Let's say row_index = 1 for the example
            # Let's say row_index = 1 for the example
            row_index = 1

            # Access the content of the column for row_index
            lazy_frame_in_row = nxs_data_eager[row_index, "/scan/data/epoch"]
            print(nxs_data_eager[row_index, "/scan/data/epoch"])
            print(type(nxs_data_eager[row_index, "/scan/data/epoch"]))

            # Check if the value is a LazyFrame before calling collect
            if isinstance(lazy_frame_in_row, pl.LazyFrame):
                # Collect the nested LazyFrame content
                resolved_data = lazy_frame_in_row.collect()
                print("Resolved data from row {}:".format(row_index), resolved_data)
            else:
                # Handle the case where the data is not a LazyFrame
                print(f"Row {row_index} does not contain a LazyFrame. It contains: {lazy_frame_in_row}")
            print(30 * "\N{pineapple}")
                        
                        

            # Resolve nested LazyDatasets if any
            if isinstance(nxs_data, pl.LazyFrame):
                nxs_data = self._resolve_nested_lazy_datasets(nxs_data)

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



    def _resolve_soft_links(self, df: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
        """
        Resolves soft links in multiple columns by mapping each row-element to the corresponding value in another column.
        Optimized to handle cases where all row elements in a column are the same.

        Args:
            df (pl.LazyFrame): The LazyFrame containing the columns.
            columns (list[str]): The column names containing the full paths (soft links).

        Returns:
            pl.LazyFrame: The LazyFrame with soft links resolved.
        """
        for column in columns:
            # Ensure the column is of string type
            df = df.with_columns(pl.col(column).cast(pl.Utf8))

            # Collect only the soft link column to check for soft links
            soft_link_values = df.select(pl.col(column)).collect().to_series()

            # Check if the column contains soft links (e.g., paths starting with '/')
            if soft_link_values.str.contains(r"^/").any():
                logger.info(f"Resolving soft links in column: {column}")
                logger.debug(f"First row of '{column}': {soft_link_values[0]}")

                # Check if all row elements in the column are the same
                unique_values = soft_link_values.unique().to_list()
                if len(unique_values) == 1:
                    # All rows point to the same target column
                    target_column = unique_values[0]
                    if target_column in df.columns:
                        logger.debug(f"All rows point to the same target column: {target_column}")
                        df = df.with_columns(pl.col(target_column).alias(column))
                    else:
                        logger.warning(f"Target column '{target_column}' not found in DataFrame.")
                        df = df.with_columns(pl.lit(None).alias(column))
                else:
                    # Rows point to different target columns
                    logger.debug(f"Rows in column '{column}' point to different target columns. Resolving row by row.")
                    path_to_value = {
                        path: pl.col(path) if path in df.columns else pl.lit(None)
                        for path in unique_values
                    }

                    df = df.with_columns(
                        pl.col(column).map_dict(path_to_value).alias(column)
                    )

        return df

    

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