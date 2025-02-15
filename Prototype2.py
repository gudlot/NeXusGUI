import streamlit as st
import polars as pl
import h5py
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from typing import Any

class GUI(ABC):
    @abstractmethod
    def render(self, browser):
        pass
    @abstractmethod
    def show_message(self, message):
        pass
    @abstractmethod
    def show_error(self, error):
        pass

class StreamlitGUI(GUI):
    def render(self, browser):
        st.title("NXS File Browser")
        
        # Custom CSS to fix truncated file names
        st.markdown(
            """
            <style>
            .stMultiSelect [data-baseweb="tag"] {
                max-width: 100%;  /* Allow tags to expand */
                white-space: nowrap;  /* Prevent wrapping */
                overflow: hidden;  /* Hide overflow */
                text-overflow: ellipsis;  /* Show ellipsis for overflow */
            }
            .stMultiSelect [data-baseweb="select"] > div {
                width: 100%;  /* Expand the dropdown width */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        # Directory input
        browser.selected_path = st.text_input("Enter directory path:", browser.root_directory)
        
        # List files button
        if st.button("List .nxs files"):
            browser.list_nxs_files()
        
        # File selection UI
        if "nxs_files" in st.session_state:
            files = st.session_state.nxs_files
            if files:
                # Select All checkbox
                select_all = st.checkbox("Select All")
                
                # File multiselect with Select All logic
                selected_files = st.multiselect(
                    "Select Nexus files:",
                    files,
                    default=files if select_all else [],
                    format_func=lambda x: x,  # Ensure full names are displayed
                    key="file_selector"  # Unique key for the widget
                )
                
                # Track changes in selected files
                if "selected_files" not in st.session_state:
                    st.session_state.selected_files = set()
                if "deleted_files" not in st.session_state:
                    st.session_state.deleted_files = set()
                
                # Add new files to the DataFrame
                new_files = set(selected_files) - st.session_state.selected_files
                if new_files:
                    for file in new_files:
                        full_path = Path(browser.selected_path) / file
                        new_data = browser.processor.process_single_file(full_path)
                        if "df" not in st.session_state:
                            st.session_state.df = pl.DataFrame([new_data])
                        else:
                            st.session_state.df = st.session_state.df.vstack(pl.DataFrame([new_data]))
                    st.session_state.selected_files.update(new_files)
                    st.session_state.deleted_files -= new_files  # Remove from deleted files if re-added
                
                # Remove deleted files from the DataFrame
                deleted_files = st.session_state.selected_files - set(selected_files)
                if deleted_files:
                    st.session_state.df = st.session_state.df.filter(
                        ~pl.col("filename").is_in(list(deleted_files))
                    )
                    st.session_state.selected_files -= deleted_files
                    st.session_state.deleted_files.update(deleted_files)  # Track deleted files
                
                # Debug info
                st.write("Selected Files:", selected_files)
                st.write("Deleted Files:", st.session_state.deleted_files)
                
                # Display the updated DataFrame
                if "df" in st.session_state and not st.session_state.df.is_empty():
                    self.display_data_overview(st.session_state.df)
                else:
                    st.write("No data to display")
            else:
                st.write("No .nxs files found in directory")

    def display_data_overview(self, df: pl.DataFrame):
        st.write("### Data Overview")

        # Show summary statistics
        st.write(f"**Total Rows:** {df.height}")
        st.write(f"**Total Columns:** {len(df.columns)}")

        # Ensure the filename column exists
        if "filename" not in df.columns:
            st.error("Filename column not found in the DataFrame.")
            return

        # Move the filename column to the front
        df = df.select(["filename"] + [col for col in df.columns if col != "filename"])

        # Convert Polars DataFrame to Pandas
        df_pd = df.to_pandas()

        # Identify 2D data columns
        two_d_columns = self._get_2d_columns(df)
        st.write("2D Columns:", two_d_columns)

        # Identify columns containing NumPy arrays
        ndarray_columns = [col for col in df_pd.columns if isinstance(df_pd[col].iloc[0], np.ndarray)]

        def truncate_array(x: Any, max_elements: int = 5) -> str:
            """Ensure formatted values are long enough to trigger tooltips."""
            if isinstance(x, np.ndarray):
                if x.ndim == 1:
                    truncated = f"[{', '.join(map(str, x[:max_elements]))}, ...]"
                    return truncated if len(truncated) > 20 else truncated + " " * 20  # Ensure overflow
                elif x.ndim == 2:
                    rows, cols = x.shape
                    truncated = f"[{', '.join(map(str, x[0, :max_elements]))}, ...]"
                    return truncated if len(truncated) > 20 else truncated + " " * 20  # Ensure overflow
            return str(x) if len(str(x)) > 20 else str(x) + " " * 20  # Force overflow

        def format_2d_array_for_tooltip(x: Any, max_rows: int = 3, max_cols: int = 5) -> str:
            if isinstance(x, np.ndarray) and x.ndim == 2:
                rows, cols = x.shape
                formatted_rows = []
                for i in range(min(rows, max_rows)):
                    row_str = ", ".join(map(str, x[i, :min(cols, max_cols)]))
                    formatted_rows.append(f"[{row_str}, ...]" if cols > max_cols else f"[{row_str}]")
                if rows > max_rows:
                    formatted_rows.append("[...]")
                return "<br>".join(formatted_rows)  # Replace \n with <br>
            return str(x)

        # Store full values as tooltips
        full_values = {col: df_pd[col].apply(lambda x: format_2d_array_for_tooltip(x) if isinstance(x, np.ndarray) and x.ndim == 2 else str(x))
                    for col in ndarray_columns}

        # Apply truncation for display (shortened values in table)
        for col in ndarray_columns:
            df_pd[col] = df_pd[col].apply(truncate_array)  

        # **Assign tooltips efficiently using pd.concat() to avoid fragmentation**
        tooltip_df = pd.DataFrame({col + "_tooltip": full_values[col] for col in ndarray_columns})
        df_pd = pd.concat([df_pd, tooltip_df], axis=1)

        # **Set up AgGrid configuration**
        gb = GridOptionsBuilder.from_dataframe(df_pd)

        # Configure columns without explicit tooltipField (forces grey tooltips)
        for col in df_pd.columns:
            if col in two_d_columns:
                gb.configure_column(col, cellStyle={"backgroundColor": "yellow", "whiteSpace": "normal"}, width=120)
            else:
                gb.configure_column(col, width=120)  # Set a reasonable width to trigger default tooltips
                    
        # Assign tooltips (full data)
        for col in ndarray_columns:
            gb.configure_column(col, tooltipField=col + "_tooltip")

        # Ensure grey tooltips appear correctly
        gb.configure_grid_options(suppressCellBrowserTooltip=False)  
        gb.configure_grid_options(domLayout="normal", tooltipShowDelay=0)

        # Enable column resizing and auto height
        gb.configure_default_column(resizable=True, wrapText=False, autoHeight=False)  # **Auto height ensures tooltips trigger**

        # Build grid options
        grid_options = gb.build()

        # **Display using AgGrid**
        grid_response = AgGrid(
            df_pd,
            gridOptions=grid_options,
            fit_columns_on_grid_load=False,
            enable_enterprise_modules=True,
            height=500,
            allow_unsafe_jscode=True,  # Required for tooltips
        )
        
    def highlight_2d_cols(s):
        return ['background-color: yellow' if s.name in two_d_columns else '' for _ in s]

    
    def _style_2d_columns(self, df_pd: pd.DataFrame, two_d_columns: list[str]) -> pd.DataFrame:
        """
        Apply background colour to 2D data columns using Pandas Styler.
        """
        def highlight_2d_cols(s):
            return ['background-color: yellow' if s.name in two_d_columns else '' for _ in s]

        return df_pd.style.apply(self.highlight_2d_cols, axis=0)
        

    def highlight_2d_columns(self, df: pd.DataFrame, two_d_columns: list[str]) -> pd.DataFrame:
        """
        Apply background color to columns containing 2D data.
        """
        # Create a style DataFrame with the same shape as the original DataFrame
        style = df.style
        
        # Apply background color to 2D columns
        for col in two_d_columns:
            style = style.apply(lambda x: ['background-color: yellow' if x.name == col else '' for i in x], axis=0)
        
        return style

    def _get_2d_columns(self, df: pl.DataFrame) -> list[str]:
        """Identify columns containing 2D data."""
        return [
            col for col in df.columns 
            if df[col].dtype == pl.List and 
            any(isinstance(row, list) and any(isinstance(i, list) for i in row) for row in df[col].to_list())
        ]

    def _get_column_config(self, df: pl.DataFrame) -> dict:
        """Generate column configuration for the DataFrame."""
        column_config = {}
        for col in df.columns:
            if df[col].dtype in (pl.Int64, pl.Float64):  # Numerical columns
                column_config[col] = self._get_number_column_config(col, df[col].dtype)
            elif df[col].dtype == pl.List:  # List columns
                column_config[col] = self._get_list_column_config(col, df[col])
            else:  # Text or other columns
                column_config[col] = self._get_text_column_config(col, df[col])
        return column_config

    def _get_number_column_config(self, col: str, dtype: pl.DataType) -> st.column_config.Column:
        """Generate configuration for numerical columns."""
        return st.column_config.NumberColumn(
            help=f"Column: {col}",
            width="medium",
            format="%d" if dtype == pl.Int64 else "%.2f"  # Format integers or floats
        )

    def _get_list_column_config(self, col: str, series: pl.Series) -> st.column_config.Column:
        """Generate configuration for list columns."""
        # Convert lists to strings using map_elements
        string_series = series.map_elements(lambda x: str(x), return_dtype=pl.String)  # Convert each list to its string representation
        max_length = string_series.str.len_chars().max()  # Calculate the maximum length
        column_width = min(max_length * 8 + 20, 300)  # Add a buffer (e.g., 20 pixels)
        return st.column_config.Column(
            help=f"Column: {col}",
            width=column_width  # Set dynamic width based on content
        )

    def _get_text_column_config(self, col: str, series: pl.Series) -> st.column_config.Column:
        """Generate configuration for text or other columns."""
        if series.dtype == pl.Utf8:  # String column
            max_length = series.str.len_chars().max()
        else:  # Non-string column (e.g., boolean, dates)
            max_length = series.cast(pl.Utf8).str.len_chars().max()
        column_width = min(max_length * 8 + 20, 300)  # Add a buffer (e.g., 20 pixels)
        return st.column_config.Column(
            help=f"Column: {col}",
            width=column_width  # Set dynamic width based on content
        )

    def _inject_custom_css(self):
        """Inject custom CSS to fix numerical values display."""
        st.markdown(
            """
            <style>
            .stDataFrame .stNumber {
                background-color: transparent !important;
                border: none !important;
                padding: 0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    
                                       
    def show_message(self, message):
        st.write(message)

    def show_error(self, error):
        st.error(error)

class NexusDataProcessor:
    def __init__(self):
        self.data = None
    
    def extract_data(self, h5_obj, path="/", data_dict=None):
        if data_dict is None:
            data_dict = {}
        for key in h5_obj.keys():
            full_path = f"{path}{key}"
            item = h5_obj[key]
            if isinstance(item, h5py.Group):
                self.extract_data(item, full_path + "/", data_dict)
            elif isinstance(item, h5py.Dataset):
                try:
                    data = item[()]
                    if isinstance(data, np.ndarray):
                        data_dict[full_path] = data.tolist()
                    elif isinstance(data, (bytes, str)):
                        data_dict[full_path] = data.decode("utf-8") if isinstance(data, bytes) else data
                    else:
                        data_dict[full_path] = str(data)
                except Exception as e:
                    data_dict[full_path] = f"Error: {e}"
        return data_dict

    def process_single_file(self, file_path: Path) -> dict:
        with h5py.File(file_path, "r") as f:
            if "scan" in f:
                data_dict = self.extract_data(f["scan"], "/scan/")
                data_dict["filename"] = file_path.name
                return data_dict
            else:
                return {"filename": file_path.name}

    def process_multiple_files(self, file_paths: list) -> pl.DataFrame:
        # Build all data in one go to avoid fragmentation
        all_data = [self.process_single_file(fp) for fp in file_paths]
        return pl.DataFrame(all_data)

class NXSFileBrowser:
    def __init__(self, gui: GUI):
        self.root_directory = "/Users/lotzegud/P08/fio_nxs_and_cmd_tool"
        self.selected_path = self.root_directory
        self.processor = NexusDataProcessor()
        self.gui = gui

    def list_nxs_files(self):
        folder = Path(self.selected_path)
        if folder.is_dir():
            files = sorted(f.name for f in folder.glob("*.nxs"))
            st.session_state.nxs_files = files
            if not files:
                self.gui.show_message("No .nxs files found")
        else:
            self.gui.show_error("Invalid directory")

    def run(self):
        self.gui.render(self)

if __name__ == "__main__":
    if "browser" not in st.session_state:
        st.session_state.browser = NXSFileBrowser(StreamlitGUI())
    st.session_state.browser.run()