import streamlit as st
import polars as pl
import h5py
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from typing import Any
from datetime import datetime
import dateutil.parser
import matplotlib.pyplot as plt
import itertools

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

        if df.shape[0] == 0:
            st.write("No data found!")
        else:
            # Convert Polars DataFrame to Pandas for easier display
            df_pd = df.to_pandas()

            # Identify 2D data columns
            two_d_columns = self._get_2d_columns(df)
            st.write("2D Columns:", two_d_columns)

            # Identify columns containing NumPy arrays
            ndarray_columns = [col for col in df_pd.columns if isinstance(df_pd[col].iloc[0], np.ndarray)]

            # Function to truncate arrays or long strings
            def truncate_array(x: Any, max_elements: int = 5) -> str:
                """Truncate arrays or long strings for display."""
                if isinstance(x, np.ndarray):
                    if x.ndim == 1:
                        truncated = f"[{', '.join(map(str, x[:max_elements]))}, ...]"
                        return truncated if len(truncated) > 20 else truncated + " " * 20  # Ensure overflow
                    elif x.ndim == 2:
                        rows, cols = x.shape
                        truncated = f"[{', '.join(map(str, x[0, :max_elements]))}, ...]"
                        return truncated if len(truncated) > 20 else truncated + " " * 20  # Ensure overflow
                elif isinstance(x, str) and len(x) > 20:  # Truncate long strings
                    return x[:20] + "..."
                return str(x) if len(str(x)) > 20 else str(x) + " " * 20  # Force overflow

            # Function to format 2D arrays for tooltips
            def format_2d_array_for_tooltip(x: Any, max_rows: int = 3, max_cols: int = 5) -> str:
                """Format 2D arrays for tooltips."""
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
            full_values = {
                col: df_pd[col].apply(lambda x: format_2d_array_for_tooltip(x) if isinstance(x, np.ndarray) and x.ndim == 2 else str(x))
                for col in ndarray_columns
            }

            # Apply truncation for display (shortened values in table)
            for col in df_pd.columns:
                df_pd[col] = df_pd[col].apply(truncate_array)

            # Assign tooltips efficiently using pd.concat() to avoid fragmentation
            tooltip_df = pd.DataFrame({col + "_tooltip": full_values[col] for col in ndarray_columns})
            df_pd = pd.concat([df_pd, tooltip_df], axis=1)

            # Set up AgGrid configuration
            gb = GridOptionsBuilder.from_dataframe(df_pd)

            # Configure columns without explicit tooltipField (forces grey tooltips)
            for col in df_pd.columns:
                if col in two_d_columns:
                    # Highlight 2D columns with yellow background
                    # Highlight 2D columns with yellow background
                    gb.configure_column(
                        col,
                        cellStyle={"backgroundColor": "yellow", "whiteSpace": "normal"},
                        width=120,
                    )
                else:
                    gb.configure_column(col, width=120)  # Set a reasonable width to trigger default tooltips

            # Assign tooltips (full data)
            for col in ndarray_columns:
                gb.configure_column(col, tooltipField=col + "_tooltip")

            # Ensure grey tooltips appear correctly
            gb.configure_grid_options(suppressCellBrowserTooltip=False)
            gb.configure_grid_options(domLayout="normal", tooltipShowDelay=0)

            # Enable column resizing and auto height
            gb.configure_default_column(resizable=True, wrapText=False, autoHeight=False)  # Auto height ensures tooltips trigger

            # Build grid options
            grid_options = gb.build()

            # Display using AgGrid
            grid_response = AgGrid(
                df_pd,
                gridOptions=grid_options,
                fit_columns_on_grid_load=False,
                enable_enterprise_modules=True,
                height=500,
                allow_unsafe_jscode=True,  # Required for tooltips
            )
            
            self.display_function(df)
                

    def display_function(self, df: pl.DataFrame):
        if "/scan/time_series_calc" not in df.columns:
            st.error("/scan/time_series_calc column not found in DataFrame.")
            return
        
        st.write("/scan/time_series_calc column found in DataFrame.")
        valid_plot_columns = [col for col in df.columns if col != "/scan/time_series_calc"]
        st.write(f"Total number of columns (excluding /scan/time_series_calc): {len(valid_plot_columns)}")
        
        unique_data_types = {type(df[col].to_list()[0]) for col in valid_plot_columns}
        st.write("Unique data types in the DataFrame:", unique_data_types)
        
        if valid_plot_columns:
            self.selected_column = st.selectbox("Select a column to plot against time:", valid_plot_columns)
            if self.selected_column:
                self.plot_columns_against_time(df, self.selected_column)

    def plot_columns_against_time(self, df: pl.DataFrame, plot_column: str):
        if "/scan/time_series_calc" not in df.columns:
            st.error("/scan/time_series_calc column is missing in DataFrame.")
            return
        
        # Ensure time periods within each row are sorted in ascending order
        df = df.with_columns(pl.col("/scan/time_series_calc").map_elements(lambda x: np.sort(x)))
        
        # Sort dataframe by the earliest timestamp in each row
        df = df.with_columns(earliest_time=pl.col("/scan/time_series_calc").list.first()).sort("earliest_time").drop("earliest_time")
        
        time_series_list = df["/scan/time_series_calc"].to_list()
        values_list = df[plot_column].to_list()
        filenames = df["filename"].to_list() if "filename" in df.columns else [f"Row {i}" for i in range(len(time_series_list))]
        
        if isinstance(values_list[0], np.ndarray) and len(values_list[0].shape) == 2:
            st.write("2D data detected. Choose visualization mode:")
            plot_mode = st.radio("Visualization Mode", ("Line Plot (Single Index)", "Heatmap"))
            
            if plot_mode == "Line Plot (Single Index)":
                index_choice = st.slider("Select index along second dimension:", 0, values_list[0].shape[1] - 1, 0)
                fig, ax = plt.subplots(figsize=(10, 5))
                
                for idx, (time_series, value, filename) in enumerate(zip(time_series_list, values_list, filenames)):
                    if index_choice < value.shape[1]:
                        ax.plot(time_series, value[:, index_choice], label=filename)
                
                ax.set_xlabel("Time")
                ax.set_ylabel(f"{plot_column}[{index_choice}]")
                ax.set_title(f"Plot of {plot_column}[{index_choice}] vs Time")
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
                ax.grid()
                fig.tight_layout()
                st.pyplot(fig)
            
            elif plot_mode == "Heatmap":
                combined_time = np.concatenate(time_series_list)
                combined_values = np.vstack(values_list)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(combined_values.T, cmap="viridis", xticklabels=False, yticklabels=False, ax=ax)
                ax.set_xlabel("Time")
                ax.set_ylabel("Index in second dimension")
                ax.set_title(f"Heatmap of {plot_column} over Time")
                st.pyplot(fig)
                
        else:
            markers = {i: marker for i, marker in zip(range(len(time_series_list)), itertools.cycle(['o', 's', 'D', '^', 'v', '<', '>']))}
            colors = {i: color for i, color in zip(range(len(time_series_list)), itertools.cycle(plt.cm.tab10.colors))}
            
            fig, ax = plt.subplots(figsize=(10, 5))
            unique_strings = {val: idx for idx, val in enumerate(sorted(set(filter(lambda x: isinstance(x, str), values_list))))}
            
            for idx, (time_series, value, filename) in enumerate(zip(time_series_list, values_list, filenames)):
                length = len(time_series)
                marker = markers[idx]
                color = colors[idx]
                
                if isinstance(value, str):
                    expanded_values = [unique_strings[value]] * length
                elif isinstance(value, np.ndarray):
                    if value.shape == (1,):
                        expanded_values = [value.item()] * length
                    elif len(value) == length:
                        expanded_values = value.tolist()
                    else:
                        st.error(f"Cannot plot {plot_column}: Invalid data format.")
                        return
                elif isinstance(value, (int, float)):
                    expanded_values = [value] * length
                else:
                    st.error(f"Cannot plot {plot_column}: Unsupported data format.")
                    return
                
                ax.plot(time_series, expanded_values, marker=marker, linestyle="-", label=filename, color=color)
            
            ax.set_xlabel("Time")
            ax.set_ylabel(plot_column)
            ax.set_title(f"Plot of {plot_column} vs Time")
            ax.grid()
            
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
            fig.tight_layout()
            
            if unique_strings:
                legend_text = "\n".join([f"{idx}: {val}" for val, idx in unique_strings.items()])
                st.text("Categorical Legend:")
                st.text(legend_text)
            
            st.pyplot(fig)

        
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
        two_d_columns = []
        
        # Known columns with 2D data
        known_2d_columns = {
            "/scan/data/amptek_spectrum": (101, 2048),
            "/scan/data/data": (101, 2048),
            "/scan/instrument/amptek/data": (101, 2048),
        }
        
        for col in df.columns:
            # Check if the column is in the known 2D columns
            if col in known_2d_columns:
                two_d_columns.append(col)
                continue  # Skip further checks for known columns
            
            # For other columns, check if they contain 2D arrays
            # Filter out null values and get the first non-null value
            non_null_values = df[col].filter(df[col].is_not_null()).to_list()
            if non_null_values:
                sample_value = non_null_values[0]  # Get the first non-null value
                if isinstance(sample_value, np.ndarray) and sample_value.ndim == 2:
                    two_d_columns.append(col)
                elif isinstance(sample_value, list) and any(isinstance(item, (list, np.ndarray)) for item in sample_value):
                    two_d_columns.append(col)
        
        return two_d_columns

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

    def extract_data(self, h5_obj: h5py.Group, path: str = "/", data_dict: dict | None = None) -> dict:
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
                        data_dict[full_path] = data
                    elif isinstance(data, (bytes, bytearray)):
                        data_dict[full_path] = data.decode("utf-8", errors="ignore")  # Avoid decoding errors
                    else:
                        data_dict[full_path] = data
                except Exception as e:
                    data_dict[full_path] = f"Error: {e}"
        
        return data_dict

    def find_nxentry(self, h5_obj, path="/"):
        """Recursively find the NXentry group dynamically."""
        for key in h5_obj.keys():
            full_path = f"{path}{key}"
            item = h5_obj[key]
            
            if isinstance(item, h5py.Group):
                if item.attrs.get("NX_class") in [b"NXentry", "NXentry"]:
                    print(f"Found NXentry: {full_path}")
                    return item, full_path
                # Recursively search in sub-groups **only if item is a group**
                result = self.find_nxentry(item, full_path + "/")
                if result[0]:
                    return result
        
        return None, None

    def process_single_file(self, file_path: Path) -> dict:
        with h5py.File(file_path, "r") as f:
            nxentry_group, nxentry_path = self.find_nxentry(f)
            if not nxentry_group:
                raise ValueError("No NXentry found in file. Ensure the file is correctly structured.")
            
            # Extract general dataset
            data_dict = self.extract_data(nxentry_group, nxentry_path + "/")
            data_dict["filename"] = file_path.name
            
            # Extract time-related data
            if "/scan/start_time" in data_dict and "/scan/end_time" in data_dict and "/scan/data/epoch" in data_dict:
                start_time_str = data_dict["/scan/start_time"]
                end_time_str = data_dict["/scan/end_time"]
                epoch = np.array(data_dict["/scan/data/epoch"], dtype=np.float64)

                # Convert to Unix timestamps
                start_time = dateutil.parser.isoparse(start_time_str).timestamp()
                end_time = dateutil.parser.isoparse(end_time_str).timestamp()

                # Use epoch as absolute timestamps
                time_series_calc = np.array(epoch, dtype=np.float64)

                # Ensure consistency with end_time
                if not np.isclose(time_series_calc[-1], end_time, atol=1e-6):
                    raise ValueError("Epoch values do not match end_time!")

                # Store time_series_calc in a list format to maintain dictionary structure
                data_dict["/scan/time_series_calc"] = time_series_calc.tolist()
                print('Length time series', len(data_dict["/scan/time_series_calc"]) )
                

        return data_dict
    
    def process_multiple_files(self, file_paths: list | Path) -> pl.DataFrame:
        if isinstance(file_paths, Path):
            file_paths = [file_paths]  # Convert to a list if it's a single Path
        
        all_data = [self.process_single_file(fp) for fp in file_paths]
        return pl.DataFrame(all_data)


    
    def list_groups(self, h5_obj, path="/"):
        """Recursively list all groups and their attributes in the file."""
        for key in h5_obj.keys():
            full_path = f"{path}{key}"
            item = h5_obj[key]

            if isinstance(item, h5py.Group):
                print(f"Group: {full_path}, Attributes: {dict(item.attrs)}")
                self.list_groups(item, full_path + "/")  # Recurse into sub-groups


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