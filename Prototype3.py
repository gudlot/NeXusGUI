import streamlit as st
from st_keyup import st_keyup
from pathlib import Path
from datetime import datetime
import polars as pl
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import logging
import re
import h5py 
from nexus_processing import NeXusBatchProcessor
from fio_processing import FioBatchProcessor
from data_controller import DataController

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="st_aggrid")


# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set the page layout to wide (MUST be the first Streamlit command)
st.set_page_config(layout="wide")

st.markdown("""
    <style>
        /* Reduce space above title but keep it visible */
        .block-container {
            padding-top: 3rem !important;  /* Ensure title remains visible */
            padding-bottom: 0.5rem !important;
        }
        /* Reduce margin around widgets */
        .stRadio, .stTextInput, .stSelectbox, .stButton, .stCheckbox, .stAgGrid, .stSlider, .stFileUploader, .stTextArea {
            margin-bottom: 0.0rem !important;
            padding-bottom: 0.0rem !important;
        }
        /* Adjust Streamlit’s main layout */
        .stApp {
            margin-top: 0rem !important;  /* No negative margin to avoid hiding title */
        }
        /* Reduce font sizes */
        html, body, .stText, .stMarkdown, .stDataFrame, .stTable {
            font
        }
        /* Adjust header sizes */
        h1 {
            font-size: 18px !important;
            margin-bottom: 0.2rem !important;
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        } 
        h2 {
            font-size: 16px !important;
            margin-bottom: 0.2rem !important;
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        } 
        h3 {
            font-size: 14px !important;
            margin-bottom: 0.2rem !important;
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        } 
        h4, h5, h6 {
            font-size: 11px !important;
            margin-bottom: 0.2rem !important;
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        } 
            </style>
""", unsafe_allow_html=True)



class FileFilterApp:
    def __init__(self, default_path: str = ""):
        """Initialize FileFilterApp with session state and processors."""    
            
        self.path = default_path
        if not self._is_valid_directory():
            logger.warning(f"Default path is invalid: {default_path}")
            self.path = ""  # Reset to empty if default path is invalid
        
        
        self.file_filter = ""
        self.extension_filter = ""
        self.selected_files = []
        self.selected_metadata = None
        self.processed_data = {}
        
        self.nxs_processor = NeXusBatchProcessor(self.path)
        self.fio_processor = FioBatchProcessor(self.path)
        self.controller = None
           
        # Initialize session state
        self._initialize_session_state()
        

    def _initialize_session_state(self):
        """Initialize session state variables if they don't already exist."""
        session_state_defaults = {
            "selected_files": [],
            "selected_metadata": None,
            "extension_filter": "All",
            "file_filter": "",
            "current_path": self.path,
        }
        
        for key, value in session_state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
    def _is_valid_directory(self, path: str = None) -> bool:
        """
        Check if the given path (or self.path if None) is set and points to a valid directory.
        Args:
            path (str, optional): The path to check. If None, uses self.path.
        Returns:
            bool: True if the path is valid, False otherwise.
        """
        path_to_check = path if path is not None else self.path
        return path_to_check and Path(path_to_check).is_dir()
    
    def _update_path(self, new_path: str):
        """
        Update the directory path and session state if the new path is valid.
        Args:
            new_path (str): The new directory path.
        Raises:
            ValueError: If the new path is invalid.
        """
        if not self._is_valid_directory(new_path):
            raise ValueError(f"Invalid directory: {new_path}")
        
        self.path = new_path
        st.session_state["current_path"] = new_path
        logger.info(f"Updating path to: {new_path}")
                
    #Flexibility: The force_reload parameter allows you to control whether the data 
    # should be reload
    @staticmethod
    @st.cache_data
    def load_nxs_files(path: str, force_reload: bool = False):
        """Load NeXus files as a lazy dataframe (cached)."""
        processor = NeXusBatchProcessor(path)
        return processor.get_lazy_dataframe(force_reload=force_reload)

    @staticmethod
    @st.cache_data
    def load_fio_files(path: str, force_reload: bool = False):
        """Load FIO files as a dataframe (cached)."""
        processor = FioBatchProcessor(path)
        return processor.get_dataframe(force_reload=force_reload)

        
    def _initialize_processors(self):
        self.nxs_processor=NeXusBatchProcessor(self.path)
        self.fio_processor = FioBatchProcessor(self.path)
        
    def _reset_app(self, new_path: str = None):
        """
        Reset the app to its initial state, optionally updating the directory path.
        
        Args:
            new_path (str, optional): The new directory path to set. If None, the current path is retained.
        """
        # Update path if a new path is provided
        if new_path:
            self._update_path(new_path)

        # Reset class attributes 
        self.file_filter = ""
        self.extension_filter = ""
        self.selected_files = []
        self.selected_metadata = None
        self.processed_data = {}
        # Reset session state to defaults
        self._initialize_session_state()

        # Clear cache and reinitialize processors
        #st.cache_data.clear() clears all cached data for functions decorated with @st.cache_data. 
        #This is a global operation and affects all cached functions, not just the ones related to your file loading.
        st.cache_data.clear()
        self._initialize_processors()
        # Clear cache and reload data using standalone functions
        nxs_df = self.load_nxs_files(self.path, force_reload=True)
        fio_df = self.load_fio_files(self.path, force_reload=True)
        
        # Update the controller
        self.controller = DataController(nxs_df, fio_df)

        # Rerun to refresh UI
        st.rerun()

         
    def run(self):
        st.title("NeXus-Fio-File Plotting App")
        
        # Load cached data using the standalone functions
        nxs_df = self.load_nxs_files(self.path)  
        fio_df = self.load_fio_files(self.path)
                

        # Initialize the controller with the DataFrames
        self.controller = DataController(nxs_df, fio_df)

        # Create two columns, with the right column wider for plotting
        col1, col2 = st.columns([2, 3])

        with col1:
            self._render_left_column()

        with col2:
            self._render_right_column()
        

    def _render_left_column(self):
        st.header("File Selection")

        # Get user input for the directory path
        new_path = st.text_input("Enter the directory path:", value=st.session_state["current_path"])
                
        # Update path only if it actually changed
        if new_path and new_path != st.session_state["current_path"]:
            try:
                # Reset the app with the new path
                self._reset_app(new_path)
            except ValueError as e:
                st.error(f"Invalid directory: {new_path}. Error: {e}")
                    
        col_reload, col_reset = st.columns([1, 1])
        with col_reload:
            if st.button("Force Reload"):
                #Keep the session state intact. I just want to refresh the underlying data.            
                # Clear the cache (this will force all cached functions to reload)
                st.cache_data.clear()
                # Force reload the data
                # Reload the data (no need for force_reload=True since the cache is cleared), #TODO: I leave this comment for the moment still here, but I could remove the force_reload=Ture actually 
                # Clearing the cache is a "sledgehammer" approach
                nxs_df = self.load_nxs_files(self.path, force_reload=True)
                fio_df = self.load_fio_files(self.path, force_reload=True)
                
                # Update the controller
                self.controller = DataController(nxs_df, fio_df)

                # Clear processed data
                self.processed_data.clear()
                st.rerun()
        with col_reset:
            if st.button("Reset App"):
                self._reset_app(st.session_state["current_path"])  # Reset using current path
                

        if self._is_valid_directory():
            
            # File type selection
            new_extension_filter = st.radio("Select file type:", ["All", ".fio", ".nxs"], horizontal=True, key="extension_filter")
            if new_extension_filter != st.session_state["extension_filter"]:
                st.session_state["extension_filter"] = new_extension_filter
                st.rerun()

            # File name filter (triggers rerun automatically on change)
            st_keyup("Filter filenames by string:", key="file_filter", debounce=100) #debounce 100ms

            
            # Display table with selectable rows
            self._render_selectable_table()

            # Render the second section for selecting data to plot
            self._render_plot_options()

    def _update_filtered_files(self):
        """Updates session state to trigger a rerun when filters change."""
        st.session_state["file_filter"] = st.session_state.get("file_filter", "")
        st.session_state["extension_filter"] = st.session_state.get("extension_filter", "")
        st.re_run()
        
    @staticmethod    
    def compute_optimal_column_widths(
        df: pl.DataFrame, 
        char_width: int = 9, 
        min_width: int = 150, 
        max_width: int = 800,
        default_buffer: float = 1.05,  # Default buffer for most columns
        filename_buffer: float = 1.2,  # Larger buffer for filename (checkboxes)
        scan_id_extra_padding: int = 48  # Extra padding for scan_id to avoid squeeze
    ) -> dict[str, int]:
        """
        Compute optimal column widths dynamically based on content length and column headers.

        Parameters:
        - df (pl.DataFrame): The Polars DataFrame containing the data.
        - char_width (int): Approximate pixel width per character.
        - min_width (int): Minimum column width.
        - max_width (int): Maximum column width.
        - default_buffer (float): Buffer for most columns (default: 1.05).
        - filename_buffer (float): Larger buffer for filename (default: 1.2).
        - scan_id_extra_padding (int): Extra pixels for scan_id to avoid squeezing.

        Returns:
        - dict: A dictionary mapping column names to optimal widths.
        """
        if df.is_empty():
            return {}

        col_widths = {}

        for col in df.columns:
            # Compute max string length directly in Polars (ignoring nulls)
            max_str_len = df[col].drop_nulls().cast(pl.Utf8).str.len_chars().max()

            # Ensure column header is also considered for width
            max_data_len = max(len(col), max_str_len)

            # Apply specific buffer factors
            if col.lower() == "filename":
                buffer = filename_buffer  # Larger buffer for filename (checkboxes)
            else:
                buffer = default_buffer  # Default buffer for all other columns

            # Special handling for 'scan_id': ensure column title is fully visible
            if col.lower() == "scan_id":
                max_data_len = max(max_data_len, len("scan_id"))  # Ensure header fits
                optimal_width = min(max(min_width, int(char_width * max_data_len * buffer) + scan_id_extra_padding), max_width)
            else:
                optimal_width = min(max(min_width, int(char_width * max_data_len * buffer)), max_width)

            col_widths[col] = optimal_width

        logger.debug(f"Computed column widths: {col_widths}")
        return col_widths

    
    def _fetch_and_combine_metadata(self, filtered_files):
        """Fetches and combines metadata for .nxs and .fio files."""
        nxs_files = [f for f in filtered_files if f.endswith(".nxs")]
        fio_files = [f for f in filtered_files if f.endswith(".fio")]

        logger.debug(f"NXS files: {len(nxs_files)}")
        logger.debug(f"FIO files: {len(fio_files)}")

        # Fetch metadata for .nxs files
        nxs_metadata = None
        if nxs_files:
            nxs_metadata = self.nxs_processor.get_core_metadata().collect()

        # Fetch metadata for .fio files
        fio_metadata = None
        if fio_files:
            fio_metadata = self.fio_processor.get_core_metadata()

        # Combine metadata into a single DataFrame
        if nxs_metadata is not None and fio_metadata is not None:
            combined_metadata = pl.concat([nxs_metadata, fio_metadata])
        elif nxs_metadata is not None:
            combined_metadata = nxs_metadata
        elif fio_metadata is not None:
            combined_metadata = fio_metadata
        else:
            st.warning("No selection data available.")
            return None

        # Filter metadata to include only relevant filenames
        combined_metadata = combined_metadata.filter(
            pl.col("filename").is_in(filtered_files)
        )

        return combined_metadata
    
    def _sort_metadata(self, metadata):
        """Sorts the metadata DataFrame by scan_id."""
        return metadata.with_columns(
            pl.col("scan_id").cast(pl.Int64, strict=False)
        ).sort("scan_id")
        
    def _display_aggrid_table(self, metadata):
        """Displays the metadata DataFrame in an AgGrid table."""
        # Convert Polars DataFrame to Pandas
        df_pd = metadata.to_pandas()

        # Configure AgGrid options
        gb = GridOptionsBuilder.from_dataframe(df_pd)
        column_widths = self.compute_optimal_column_widths(metadata)
        for col, width in column_widths.items():
            gb.configure_column(col, width=width, wrapText=True, autoHeight=True)
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
        grid_options = gb.build()
        grid_options["defaultColDef"] = {
            "cellStyle": {"font-size": "10px"}
        }
        # Display AgGrid table
        grid_response = AgGrid(df_pd, gridOptions=grid_options, height=400, fit_columns_on_grid_load=True)

        # Log grid response for debugging
        logger.debug("Grid Response Data: %s", grid_response.data)
        logger.debug("Grid Selected Rows: %s", grid_response.selected_rows)

        return grid_response
    
    def _update_selected_files_and_metadata(self, grid_response):
        """Updates the selected files and metadata in session state."""
        if grid_response and "selected_rows" in grid_response:
            selected_df = pd.DataFrame(grid_response["selected_rows"])
            if "filename" in selected_df.columns:
                st.session_state["selected_files"] = selected_df["filename"].tolist()
                self.selected_files = st.session_state["selected_files"]
                self.selected_metadata = pl.from_pandas(selected_df)
                st.session_state["selected_metadata"] = self.selected_metadata
            else:
                self.selected_metadata = None
                st.session_state["selected_metadata"] = None

        # Log selected files for debugging
        logger.debug(f"Selected files: {self.selected_files}")
           
        
    def _render_selectable_table(self):
        """Displays a scrollable, multi-column table with selectable rows using st_aggrid."""
        
        # Get filtered files dynamically
        filtered_files = self._get_filtered_files()
        if not filtered_files:
            st.write("No files match the filter.")
            return
        
        # Fetch and combine metadata for filtered files
        combined_metadata = self._fetch_and_combine_metadata(filtered_files)
        if combined_metadata is None:
            st.warning("No selection data available.")
            return

        # Sort metadata by scan_id
        combined_metadata = self._sort_metadata(combined_metadata)

        # Display the table using AgGrid
        grid_response = self._display_aggrid_table(combined_metadata)

        # Update selected files and metadata in session state
        self._update_selected_files_and_metadata(grid_response)
            
       
        
    #don't cache this function with st.cache_data because we expect directory content to change frequently    
    def _list_files_in_directory(self) -> list:
        """Returns a list of files in the directory."""
        path = Path(self.path).resolve()  # Ensure absolute path
        return [f.name for f in path.glob("*") if f.is_file()]
    
    def _apply_extension_filter(self, files: list) -> list:
        """Filters files based on the selected extension."""
        ext_filter = st.session_state["extension_filter"]
        valid_extensions = (".fio", ".nxs") if ext_filter == "All" else (ext_filter,)
        return [f for f in files if Path(f).suffix in valid_extensions]
    
    def _apply_filename_filter(self, files: list) -> list:
        """Filters files based on the filename filter."""
        file_filter = st.session_state["file_filter"].strip()
        if not file_filter:
            return files
        
        # Build regex pattern based on the filter
        if file_filter.isdigit():  # Match scan number
            regex_pattern = rf"(?<!\d){file_filter}(?!\d)"
        else:  # Match general filename pattern
            regex_pattern = re.escape(file_filter).replace(r"\*", ".*").replace(r"\?", ".")
        
        return [f for f in files if re.search(regex_pattern, f, re.IGNORECASE)]


    def _get_filtered_files(self):
        """Returns a list of .fio or .nxs files in the directory that match the filters."""
        
        # Check if the directory is valid
        if not self._is_valid_directory():
            st.error(f"Invalid directory: {self.path}")
                  
        logger.debug(f"Current directory path: {self.path}")

        # Get files in the directory
        files = self._list_files_in_directory()
        
        # Apply extension filter
        files = self._apply_extension_filter(files)
        
        # Apply filename filter
        files = self._apply_filename_filter(files)
        
        # Log the filtered files
        logger.debug(f"Filtered files: {files}")
        
        return files

        
    
    @staticmethod
    @st.cache_data
    def _get_column_names(selected_files, path):
        column_names = {}

        if selected_files:
            nxs_files = [f for f in selected_files if f.endswith(".nxs")]
            fio_files = [f for f in selected_files if f.endswith(".fio")]

            if nxs_files:
                nexus_processor = NeXusBatchProcessor(path)
                df_nxs = nexus_processor.get_lazy_dataframe(nxs_files)
                #The |= operator in Python is a shorthand for merging dictionaries, introduced in Python 3.9.
                # It is functionally equivalent to dict.update() but preserves ordering.
                #column_names |= {col: None for col in df_nxs.schema.keys()}
                column_names |= {col: None for col in df_nxs.collect().columns}

            if fio_files:
                fio_processor = FioBatchProcessor(path)
                df_fio = fio_processor.get_dataframe(fio_files)
                column_names |= {col: None for col in df_nxs.collect().columns}


        return column_names  # Ensuring order is preserved


    def _render_plot_options(self):
        st.header("Plot Options")

        if self.selected_files:
            time_column = "/scan/data/epoch"
            time_options = [f"{time_column} (Unix)", f"{time_column} (Datetime)"]

            # Pass required arguments to the static method
            column_names = self._get_column_names(self.selected_files, self.path)
            column_names.pop(time_column, None)  # Remove time_column directly

            col1, col2 = st.columns([3, 1])

            with col1:
                x1 = st.selectbox("Select X1 (Time):", time_options, key="x1")
                x2 = st.selectbox("Select X2:", list(column_names.keys()), key="x2")
                y1 = st.selectbox("Select Y1-axis:", list(column_names.keys()), key="y1")
                z = st.selectbox("Select Normalization Column:", list(column_names.keys()), key="z")

            with col2:
                x1_radio = st.radio("", ["Use", "Ignore"], key="x1_radio")
                x2_radio = st.radio("", ["Use", "Ignore"], key="x2_radio")
                y1_radio = st.radio("", ["Use", "Ignore"], key="y1_radio")
                z_radio = st.radio("", ["Use", "Ignore"], key="z_radio", index=1)  # Default to Ignore

            normalize = st.checkbox("Normalize data", value=False)

            if st.button("Plot"):
                x_axis = x1 if x1_radio == "Use" else (x2 if x2_radio == "Use" else None)
                y_axis = y1 if y1_radio == "Use" else None
                z_axis = z if z_radio == "Use" else None

                if not x_axis or not y_axis:
                    st.error("You must select either X1 or X2 and Y1 for plotting.")
                    return

                self.plot_data(x_axis, y_axis, z_axis, normalize)




    def plot_data(selected_files, combined_metadata, x_column, y_columns, z_column=None):
        """
        Plots data based on selected files and column types.
        
        Parameters:
        - selected_files: List of selected file indices.
        - combined_metadata: DataFrame containing metadata.
        - x_column: Column name for x-axis.
        - y_columns: List of column names for y-axis.
        - z_column: Optional column for normalization.
        """
        
        # Ensure selected files are sorted by scan_id if possible
        try:
            combined_metadata = combined_metadata.sort_values("scan_id", key=lambda x: x.astype(int))
        except ValueError:
            logging.warning("scan_id could not be converted to integers. Sorting lexicographically.")
            combined_metadata = combined_metadata.sort_values("scan_id")
        
        # Assign markers and colors
        markers = {i: marker for i, marker in zip(selected_files, itertools.cycle(['o', 's', 'D', '^', 'v', '<', '>']))}
        colors = {i: color for i, color in zip(selected_files, itertools.cycle(plt.cm.tab10.colors))}
        
        plt.figure(figsize=(10, 6))
        
        for idx in selected_files:
            row = combined_metadata.iloc[idx]
            x_data = row[x_column]
            
            if isinstance(x_data, str) or np.isscalar(x_data):
                x_data = np.array([x_data])  # Broadcast scalars/strings
            
            for y_column in y_columns:
                y_data = row[y_column]
                
                if isinstance(y_data, str) or np.isscalar(y_data):
                    y_data = np.full_like(x_data, y_data, dtype=object)  # Broadcast scalars
                
                if isinstance(y_data, np.ndarray):
                    if y_data.ndim == 1:
                        plt.plot(x_data, y_data, marker=markers[idx], color=colors[idx], label=f"{row['filename']} - {y_column}")
                    elif y_data.ndim == 2:
                        logging.debug(f"2D data detected in {y_column}. Using imshow.")
                        plt.figure()
                        plt.imshow(y_data, aspect='auto', cmap='viridis')
                        plt.colorbar(label=y_column)
                        plt.title(f"{row['filename']} - {y_column}")
                        plt.show()
                
        plt.xlabel(x_column)
        plt.ylabel(", ".join(y_columns))
        plt.legend()
        plt.title("Data Plot")
        plt.show()


    def _render_right_column(self):
        st.header("Plotting Area")
        if self.plot_data:
            # Render the plot here
            pass
        else:
            st.write("No data to plot yet.")
            
 
    def _process_time_column(self, column: str):
        if column == "/scan/data/epoch (Datetime)":
            return self._convert_unix_to_datetime("/scan/data/epoch")
        return column

    def _convert_unix_to_datetime(self, column: str):
        for file in self.selected_files:
            if file.endswith(".nxs"):
                processor = NexusProcessor(file)
                df = processor.get_dataframe()
                if column in df.columns:
                    df = df.with_columns(pl.col(column).map(lambda x: datetime.datetime.utcfromtimestamp(x)))
                return df.columns
        return column
 
 

if __name__ == "__main__":
    app = FileFilterApp(default_path="/Users/lotzegud/P08/fio_nxs_and_cmd_tool/")
    app.run()