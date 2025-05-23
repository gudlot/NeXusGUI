# NeXusGUI – A GUI for visualising data across multiple NeXus files.
# Copyright (C) 2025 Deutsches Elektronen-Synchrotron DESY
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
__copyright__ = "Deutsches Elektronen-Synchrotron DESY, Hamburg, Germany"
__date__ = "25/04/2025"
__license__ = "AGPL-3.0"
__status__ = "Development"


import streamlit as st
from st_keyup import st_keyup
from pathlib import Path
from datetime import datetime
import polars as pl
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import logging
import re

import numpy as np
from nexus_processing import NeXusBatchProcessor
from fio_processing import FioBatchProcessor
from data_controller import DataController


import matplotlib as mpl
import matplotlib.pyplot as plt
# Set the default font to DejaVu Sans
plt.rcParams['font.family'] = 'DejaVu Sans'

import matplotlib as mpl
print(mpl.get_cachedir())

# Suppress Matplotlib debug logging (including font matching)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="st_aggrid")


TIME_COLUMN = "/scan/data/epoch"




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



import polars as pl
import random
from datetime import datetime, timedelta

import re
import polars as pl
import random
from datetime import datetime, timedelta

def generate_fake_fio_metadata(fio_files):
    """Generates fake metadata for .fio files."""
    if not isinstance(fio_files, list):
        raise ValueError("fio_files must be a list")

    num_files = len(fio_files)

    # Extract scan_id from filenames (last number before .fio)
    scan_ids = []
    for filename in fio_files:
        # Use regex to find the last number before .fio
        match = re.search(r"(\d+)\.fio$", filename)
        if match:
            scan_ids.append(match.group(1))  # Extract the numeric part
        else:
            # If no number is found, generate a random scan_id
            scan_ids.append(str(random.randint(1000, 9999)))

    data = {
        "filename": fio_files,  # Length: num_files
        "scan_id": scan_ids,  # Use extracted scan_ids
        "scan_command": [f"Command_{i+1}" for i in range(num_files)],  # Match length  
        "human_start_time": [
            (datetime.now() - timedelta(days=random.randint(1, 30)))  
            .strftime('%Y-%m-%d %H:%M:%S')  
            for _ in range(num_files)
        ],  # Match length
    }
    
    # Create DataFrame
    df = pl.DataFrame(data)

    # Ensure correct types
    df = df.with_columns([
        pl.col("filename").cast(pl.Utf8),
        pl.col("scan_id").cast(pl.Utf8),
        pl.col("scan_command").cast(pl.Utf8),
        pl.col("human_start_time").cast(pl.Utf8),
    ])

    return df

class FileFilterApp:
    def __init__(self, default_path: str = ""):
        """Initialize FileFilterApp with session state and processors."""    
            

        # Initialize session state if it doesn't exist
        if "current_path" not in st.session_state:
            st.session_state["current_path"] = default_path
        
        # Use the session state path instead of the default path
        self.path = st.session_state["current_path"]
            
        if not self._is_valid_directory():
            logger.warning(f"Default path is invalid: {default_path}")
            self.path = self._prompt_for_valid_path()  # Prompt user for a valid path
            
        
        self.file_filter = ""
        self.extension_filter = ""
        self.selected_files = []
        self.selected_metadata = pd.DataFrame()
        self.processed_data = {}
        self.nxs_df = None
        self.fio_df = None 
        
        self.nxs_processor = NeXusBatchProcessor(self.path)
        self.fio_processor = FioBatchProcessor(self.path)
        self.controller = None
           
        # Initialize session state
        self._initialize_session_state()
        

    def _initialize_session_state(self):
        """Initialize session state variables if they don't already exist."""
        session_state_defaults = {
            "selected_files": [],
            "selected_metadata": pd.DataFrame(),
            "extension_filter": "All",
            "file_filter": "",
            "selected_rows":  pd.DataFrame()  # Initialize selected_rows in session state
        }
        
        for key, value in session_state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
        # Initialize current_path only if it doesn't already exist
        if "current_path" not in st.session_state:
            st.session_state["current_path"] = self.path
            
    def _prompt_for_valid_path(self) -> str:
        """Prompt the user to enter a valid directory path."""
        st.warning("The provided path is invalid. Please enter a valid directory path.")
        new_path = st.text_input("Enter a valid directory path:", value="")
        
        if new_path and self._is_valid_directory(new_path):
            return new_path
        else:
            st.error("Invalid path. Please try again.")
            return ""  # Return empty string if no valid path is provided
                    
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
        try:
            if not self._is_valid_directory(new_path):
                raise ValueError(f"Invalid directory: {new_path}")
            
            self.path = new_path
            st.session_state["current_path"] = new_path
            logger.info(f"Updating path to: {new_path}")
        except ValueError as e:
            logger.error(f"Failed to update path: {e}")
            st.error(f"Invalid directory: {new_path}. Please enter a valid path.")
                
    #Flexibility: The force_reload parameter allows you to control whether the data 
    # should be reload
    #See for st.cache_resource
    #https://docs.streamlit.io/develop/concepts/architecture/caching#stcache_resource
    @staticmethod
    @st.cache_resource
    def load_nxs_files(path: str, force_reload: bool = False):
        """Load NeXus files as a lazy dataframe (cached)."""
        processor = NeXusBatchProcessor(path)
        df =  processor.get_dataframe(force_reload=force_reload)
        print(type(df))
        
        
        return df 
        
    @staticmethod
    @st.cache_data
    def load_fio_files(path: str, force_reload: bool = False):
        """Load FIO files as a dataframe (cached)."""
        processor = FioBatchProcessor(path)
        return processor.get_dataframe(force_reload=force_reload)

        
    def _initialize_processors(self):
        """Reinitialize file processors for the current path."""
        self.nxs_processor = NeXusBatchProcessor(self.path)
        self.fio_processor = FioBatchProcessor(self.path)

        return self.nxs_processor, self.fio_processor

        
    def _reset_app(self, new_path: str = None):
        """
        Reset the app to its initial state, optionally updating the directory path.
        
        Args:
            new_path (str, optional): The new directory path to set. If None, the current path is retained.
        """
        
        try:
            # Update path if a new path is provided
            if new_path:
                self._update_path(new_path)  # Use _update_path to handle validation and updates
        except ValueError as e:
            st.error(str(e))  # Display the error message if the path is invalid
            return


        # Reset class attributes 
        self.file_filter = ""
        self.extension_filter = ""
        self.selected_files = st.session_state.get("selected_files", [])  # Reinitialize from session state
        self.selected_metadata = st.session_state.get("selected_metadata", None)  # Reinitialize from session state
        self.processed_data = {}
        # Reset session state to defaults
        self._initialize_session_state()

        # Clear cache and reinitialize processors
        #st.cache_data.clear() clears all cached data for functions decorated with @st.cache_data. 
        #This is a global operation and affects all cached functions, not just the ones related to your file loading.
        st.cache_data.clear()
        #Reinitialize processors and use them**
        self.nxs_processor, self.fio_processor = self._initialize_processors()

        #Ensure force reload actually loads new data**
        self.nxs_df = self.nxs_processor.get_lazy_dataframe(force_reload=True)
        self.fio_df = self.fio_processor.get_dataframe(force_reload=True)
           
        
        # Update the controller
        self.controller = DataController(self.nxs_df, self.fio_df, self.nxs_processor, self.fio_processor)

        
        logger.debug(75 * "\N{T-Rex}")
        logger.debug(f"Path after reset: {self.path}")  # Confirm the path is updated
        logger.debug(f"Selected files after reset: {self.selected_files}")  # Log selected files
        logger.debug(75 * "\N{T-Rex}")
            
        # Rerun to refresh UI
        st.rerun()

         
    def run(self):
        st.title("NeXus-Fio-File Plotting App")
        
        # Load cached data using the standalone functions
        self.nxs_df = self.load_nxs_files(st.session_state["current_path"])  
        self.fio_df = self.load_fio_files(st.session_state["current_path"])        

        # Initialize the controller with the DataFrames
        self.controller = DataController(self.nxs_df, self.fio_df, self.nxs_processor, self.fio_processor)
        
        st.write(isinstance(self.nxs_df, pl.LazyFrame))
                     

        # Create two columns, with the right column wider for plotting
        col1, col2 = st.columns([2, 3])

        with col1:
            self._render_left_column()

        with col2:
            self._render_right_column()
       
    # Define the callback function to reset the app when the path changes
    #@staticmethod
    def _on_path_change(self):
        self._reset_app(st.session_state["path_input"])    
        
    @staticmethod    
    def _on_extension_filter_change():
        """Callback function to handle changes in the extension filter."""
        # Update the session state with the new filter value
        st.session_state["extension_filter"] = st.session_state["extension_filter_widget"]
        st.session_state["selected_rows"] = pd.DataFrame() 
        #st.rerun()

    def _render_left_column(self):
        st.header("File Selection")
        
        
        # Get user input for the directory path with an `on_change` event
        new_path=st.text_input(
            "Enter the directory path:", 
            value=st.session_state.get("current_path", self.path), 
            key="path_input",
            on_change=self._on_path_change
        )
        
                
        # Update path only if it actually changed
        if new_path and new_path != st.session_state.get("current_path", self.path):
            try:
                # Reset the app with the new path
                self._reset_app(new_path)
            except ValueError as e:
                st.error(f"Invalid directory: {new_path}. Error: {e}")
        
             
        st.text_input("\N{hot pepper} Current Path:", st.session_state.get("current_path", ""), disabled=True)

                    
        col_reload, col_reset = st.columns([1, 1])
        with col_reload:
            if st.button("Force Reload"):
                #Keep the session state intact. I just want to refresh the underlying data.            
                # Clear the cache (this will force all cached functions to reload)
                st.cache_data.clear()
                # Force reload the data
                # Reload the data (no need for force_reload=True since the cache is cleared), #TODO: I leave this comment for the moment still here, but I could remove the force_reload=Ture actually 
                # Clearing the cache is a "sledgehammer" approach
                self.nxs_df = self.load_nxs_files(self.path, force_reload=True)
                self.fio_df = self.load_fio_files(self.path, force_reload=True)
                
                # Update the controller
                self.controller = DataController(self.nxs_df, self.fio_df)

                # Clear processed data
                self.processed_data.clear()
                logger.debug(f"Force Reload triggered")
                st.rerun()
        with col_reset:
            if st.button("Reset App"):
                self._reset_app(st.session_state["current_path"])  # Reset using current path
                

        if self._is_valid_directory():
            
            # File type selection
            #new_extension_filter = st.radio("Select file type:", ["All", ".fio", ".nxs"], horizontal=True, key="extension_filter")
            #if new_extension_filter != st.session_state["extension_filter"]:
            #    st.session_state["extension_filter"] = new_extension_filter
            #    st.rerun()

            # File type selection with on_change callback
            new_extension_filter = st.radio(
                "Select file type:",
                ["All", ".fio", ".nxs"],
                horizontal=True,
                key="extension_filter_widget",
                on_change=self._on_extension_filter_change  # Callback to update session state
            )


            # File name filter (triggers rerun automatically on change)
            st_keyup("Filter filenames by string:", key="file_filter", debounce=100) #debounce 100ms

            
            # Display table with selectable rows
            self._render_selectable_table()

            # Render the second section for selecting data to plot
            self._render_plot_options()

        
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

        #logger.debug(f"Computed column widths: {col_widths}")
        return col_widths

    
    def _fetch_and_combine_metadata(self, filtered_files):
        """Fetches and combines metadata for .nxs and .fio files."""
        nxs_files = [f for f in filtered_files if f.endswith(".nxs")]
        fio_files = [f for f in filtered_files if f.endswith(".fio")]

        logger.debug(f"NXS files: {len(nxs_files)}")
        logger.debug(f"FIO files: {len(fio_files)}")

        # Fetch metadata for .nxs files from cached data frame
        nxs_metadata = None
        if nxs_files and hasattr(self, "nxs_df"):
            nxs_metadata = (
                self.nxs_df
                .filter(pl.col("filename").is_in(nxs_files))
                .select(["filename", "scan_id", "scan_command", "human_start_time"])
                .collect()  # Collect only after filtering to minimize memory usage
            )
        logger.debug(f'nxs_metadata is {type(nxs_metadata)}')

        # Fetch metadata for .fio files from cached data frame
        fio_metadata = None
        #if fio_files and hasattr(self, "fio_df"):
        #    fio_metadata = self.fio_df.filter(
        #        pl.col("filename").is_in(fio_files)
        #    ).select(["filename", "scan_id", "scan_command", "human_start_time"])
        if fio_files:
            fio_metadata = generate_fake_fio_metadata(fio_files)
            #logging.debug(f'Fio_metadata:\n {fio_metadata}')

        # Fetch metadata for .nxs files
        #nxs_metadata = None
        #if nxs_files:
        #    nxs_metadata = self.nxs_processor.get_core_metadata().collect()
        # Fetch metadata for .fio files
        #fio_metadata = None
        #if fio_files:
        #    fio_metadata = self.fio_processor.get_core_metadata()
        
        
        #if nxs_metadata is not None:
        #    logger.debug(f"NXS Metadata Filenames: {nxs_metadata['filename'].to_list()}")
        #if fio_metadata is not None:
        #    logger.debug(f"FIO Metadata Filenames: {fio_metadata['filename'].to_list()}")
                
        #if nxs_metadata is not None:
        #    logger.debug(f"NXS Metadata Schema: {nxs_metadata.schema}")
        #if fio_metadata is not None:
        #    logger.debug(f"FIO Metadata Schema: {fio_metadata.schema}")



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
                
        logging.debug(f"Selection extension: {st.session_state.extension_filter}")
        
        logging.debug(30 * "\N{rainbow}")
        # Combine metadata into a single DataFrame
        if nxs_metadata is not None and fio_metadata is not None:
            logger.debug("Both NXS and FIO metadata are available.")
            combined_metadata = pl.concat([nxs_metadata, fio_metadata])
            logger.debug(f"Combined Metadata (NXS + FIO):\n{combined_metadata}")
        elif nxs_metadata is not None:
            logger.debug("Only NXS metadata is available.")
            logger.debug(f"NXS Metadata:\n{nxs_metadata}")
            combined_metadata = nxs_metadata
            logger.debug(f"Combined Metadata (NXS only):\n{combined_metadata}")
        elif fio_metadata is not None:
            logger.debug("Only FIO metadata is available.")
            logger.debug(f"FIO Metadata:\n{fio_metadata}")
            combined_metadata = fio_metadata
            logger.debug(f"Combined Metadata (FIO only):\n{combined_metadata}")
        else:
            logger.warning("No selection data available.")
            st.warning("No selection data available.")
            return None
        logging.debug(30 * "\N{rainbow}")
                
        return combined_metadata
    
    def _sort_metadata(self, metadata):
        """Sorts the metadata DataFrame by scan_id."""
        return metadata.with_columns(
            pl.col("scan_id").cast(pl.Int64, strict=False)
        ).sort("scan_id")
        
    def _display_aggrid_table(self, metadata, grid_key):
        """Displays the metadata DataFrame in an AgGrid table."""
        # Convert Polars DataFrame to Pandas
        df_pd = metadata.to_pandas()

        # Configure AgGrid options
        gb = GridOptionsBuilder.from_dataframe(df_pd)
        column_widths = self.compute_optimal_column_widths(metadata)
        for col, width in column_widths.items():
            gb.configure_column(col, width=width, wrapText=True, autoHeight=True, checkboxSelection=(col == "filename"),
                                editable=False,sortable=True, fitlerable=True)

        gb.configure_selection(
            selection_mode="multiple",        # Enable multiple row selection
            use_checkbox=True,                # Use checkboxes for selection
            rowMultiSelectWithClick=True,     # Allow Shift/Ctrl (Cmd) selection
            suppressRowDeselection=False,     # Allow deselection by clicking again
            suppressRowClickSelection=False   # Allow selection by clicking anywhere in the row
        )

        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
        grid_options = gb.build()
        grid_options["defaultColDef"] = {
            "cellStyle": {"font-size": "10px"}
        }
        # Display AgGrid table
        
                   
        try:
            grid_response = AgGrid(
                df_pd, 
                gridOptions=grid_options, 
                update_mode=GridUpdateMode.SELECTION_CHANGED,  # Update when selection changes 
                height=400, 
                fit_columns_on_grid_load=True,
                key=grid_key,
                reload_data=False,
                data_return_mode="FILTERED_AND_SORTED",  # Ensure selected rows are returned
         
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.exception("AgGrid failed")  # Logs full traceback

        # Update selected rows in session state based on AgGrid response
        st.session_state.selected_rows = grid_response["selected_rows"]
        st.write("Selected Rows:", st.session_state.selected_rows)  # Display selected rows
        st.write('Event triggered ', grid_response.event_data)
        #st.write("Grid options:", grid_options)

        # Log grid response for debugging
        logger.debug("Grid State: %s", grid_response.grid_state)
        logger.debug("Grid Selected Rows: %s", grid_response.selected_rows)
        logger.debug("Type grid selected rows: %s", type(grid_response.selected_rows))
            
        return grid_response
    
    def _update_selected_files_and_metadata(self, grid_response):
        """Updates the selected files and metadata in session state."""
        if grid_response and 'selected_rows' in grid_response:
            # Handle case where selected_rows is None or an empty DataFrame
            selected_rows = grid_response['selected_rows']
            if selected_rows is None or (hasattr(selected_rows, 'empty') and selected_rows.empty):
                selected_rows = []  # Use empty list if None or empty DataFrame
            else:
                selected_rows = selected_rows  # Use the DataFrame as-is

            # Extract 'filename' values from the selected rows
            if isinstance(selected_rows, list):
                selected_files = [row['filename'] for row in selected_rows]
            else:
                # If selected_rows is a DataFrame, extract the 'filename' column
                selected_files = selected_rows['filename'].tolist()

            # Update session state
            st.session_state['selected_files'] = selected_files
            self.selected_files = selected_files

            # Convert selected rows to a Polars DataFrame
            if isinstance(selected_rows, list):
                selected_df = pl.DataFrame(selected_rows)
            else:
                selected_df = pl.DataFrame(selected_rows.to_dict('records'))

            self.selected_metadata = selected_df
            st.session_state['selected_metadata'] = self.selected_metadata



            # Log selected files for debugging
            logger.debug(f"\N{hot pepper} Selected files: {self.selected_files}")
            

           
        
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
        
        # Generate dynamic key based on selected dataset (all, fio, nxs)
        dataset_key = st.session_state.get("extension_filter", "all")  # Default to 'all' if not set
        grid_key = f"aggrid_{dataset_key}"  # Ensures AgGrid resets when dataset changes

        # Add debug logging to check selected_rows state
        logger.debug(f"\N{peacock}Selected rows before AgGrid render: {st.session_state.selected_rows}")
        logger.debug('Grid key: %s', grid_key)

        try:
            # Display the table using AgGrid, pass selected_rows as initial selection if present
            grid_response = self._display_aggrid_table(combined_metadata, grid_key)
        except Exception as e:
            logger.error(f"An error occurred while rendering the table: {e}")
            grid_response = None
            
        # Check if grid_response is valid and not None
        if grid_response is not None and 'selected_rows' in grid_response:
            # If grid response has selected rows, update session state
            if grid_response.selected_rows is not None:
                st.session_state.selected_rows = grid_response.selected_rows
                logger.debug(f"🦚Updated session state selected rows: {st.session_state.selected_rows}")
        else:
            logger.debug("Grid response is invalid or doesn't contain selected rows.")
        
                
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
        
        # Define valid extensions based on the selected filter
        if ext_filter == "All":
            valid_extensions = (".fio", ".nxs")
        else:
            valid_extensions = (ext_filter.lower(),)  # Ensure lowercase for comparison
        
        # Log the filter and valid extensions for debugging
        logger.debug(f"Applying extension filter: {ext_filter}")
        logger.debug(f"Valid extensions: {valid_extensions}")
        
        # Filter files based on extension (case-insensitive)
        filtered_files = [f for f in files if Path(f).suffix.lower() in valid_extensions]
        
        # Log the filtered files for debugging
        #logger.debug(f"Filtered files: {filtered_files}")
        
        return filtered_files
    
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

        # Get files in the directory
        files = self._list_files_in_directory()
        
        # Apply extension filter
        files = self._apply_extension_filter(files)
        
        # Apply filename filter
        files = self._apply_filename_filter(files)
        
        # Log the filtered files
        logger.debug(f"Filtered files: {files}")
        
        return files

    def _render_plot_options(self):
        st.header("Plot Options")

        if self.selected_files:
            # Define the actual column name and time options
            time_column = TIME_COLUMN
            time_options = ["Unix", "Datetime"]  # Display options for time format

            # Ensure we don’t modify cached data in place
            column_names = self.controller.get_column_names(self.selected_files).copy()          
            column_names.pop(time_column, None)  # Remove time_column safely

            if not column_names:
                st.warning("No valid columns found for plotting.")
                return

            col1, col2 = st.columns([3, 1])

            with col1:
                # Display time format options (Unix or Datetime)
                time_format = st.selectbox("Select time format:", time_options, key="plot_option_t_format")
                x = st.selectbox("Select x:", list(column_names.keys()), key="plot_option_x")
                y = st.selectbox("Select y:", list(column_names.keys()), key="plot_options_y")
                z = st.selectbox("Select Normalization:", list(column_names.keys()), key="plot_options_z")

            with col2:
                t_radio = st.radio(" ", ["Use", "Ignore"], key="plot_options_t_radio", label_visibility="hidden")
                x_radio = st.radio(" ", ["Use", "Ignore"], key="plot_options_x_radio", index=1, label_visibility="hidden")
                y_radio = st.radio(" ", ["Use", "Ignore"], key="plot_options_y_radio", label_visibility="hidden")
                z_radio = st.radio(" ", ["Use", "Ignore"], key="plot_options_z_radio", index=1, label_visibility="hidden")  # Default to Ignore

            normalize = st.checkbox("Normalize data", value=False, key="plot_options_normalize")

            if st.button("Plot", key="plot_options_plot"):
                # Use the actual column name for time
                x_axis = x if x_radio == "Use" else (time_column if t_radio == "Use" else None)
                y_axis = y if y_radio == "Use" else None
                z_axis = z if z_radio == "Use" else None

                if not x_axis or not y_axis:
                    st.error("You must select both an x-axis (either x or t) and a y-axis for plotting.")
                    return
                
                logger.debug(f"Selected x-axis: {x_axis}")
                logger.debug(f"Selected y-axis: {y_axis}")
                logger.debug(f"Selected z-axis: {z_axis}")
                logger.debug(f"Selected time format: {time_format}")

                # Pass the time format to the plotting function
                self.plot_data(x_axis, y_axis, z_axis, normalize, time_format)    
    
    def plot_data(self, x_axis: str, y_axis: str, z_axis: str = None, normalize: bool = False, time_format: str = "Unix"):
        """
        Plots data based on the selected columns and files.
        """
        if not self.selected_files:
            st.warning("No files selected for plotting.")
            return

        try:
            # Retrieve data for the selected files and columns
            df_plot = self.controller.process_selected_files(
                self.selected_files, x_axis, y_axis, z_axis, normalize
            )

            print("DataFrame df_plot Structure:")
            print(df_plot.head())  # Print the first few rows of the DataFrame
            print("\nDataFrame Columns:")
            print(df_plot.columns)  # Check the columns
            print("\nDataFrame Types:")
            print(df_plot.schema)  # Check data types of the columns

            # Convert time column if necessary
            if x_axis == TIME_COLUMN and time_format == "Datetime":
                df_plot = df_plot.with_columns(pl.col(TIME_COLUMN).cast(pl.Datetime).alias(TIME_COLUMN))

            # Format and broadcast data using DataController's method
            df_plot = self.controller._format_and_broadcast_data(df_plot, x_axis, y_axis, z_axis if normalize else None)

            # Convert Polars DataFrame to Pandas for plotting
            df_plot_pd = df_plot.to_pandas()

            # Store the plot data for later use
            self.plot_data_dict = {  # Renamed to avoid conflict
                'data': df_plot_pd,
                'x_axis': x_axis,
                'y_axis': y_axis,
                'normalize': normalize
            }
            
              # Debugging: Inspect the DataFrame structure
            print("DataFrame End of plot_data Structure:")
            print(df_plot_pd.head())  # Print the first few rows of the DataFrame
            print("\nDataFrame Columns:")
            print(df_plot_pd.columns)  # Check the columns
            print("\nDataFrame Types:")
            print(df_plot_pd.dtypes)  # Check data types of the columns


            # Generate the plot
            self._generate_plot(df_plot_pd, x_axis, y_axis, normalize)

        except Exception as e:
            st.error(f"An error occurred while plotting: {e}")
            logger.error(f"Plotting error: {e}", exc_info=True)

    def _generate_plot(self, data: pd.DataFrame, x_axis: str, y_axis: str, normalize: bool = False):
        """
        Generates a plot using Matplotlib with unique markers and colors per file.

        Args:
            data (pd.DataFrame): The data to plot.
            x_axis (str): The column name for the x-axis.
            y_axis (str): The column name for the y-axis.
            normalize (bool): Whether to normalize the y-axis data.
        """
        # Debugging: Inspect the DataFrame structure
        print("DataFrame Structure:")
        print(data.head())  # Print the first few rows of the DataFrame
        print("\nDataFrame Columns:")
        print(data.columns)  # Check the columns
        print("\nDataFrame Types:")
        print(data.dtypes)  # Check data types of the columns

        # Debugging: Check the unique filenames
        unique_filenames = data["filename"].unique()
        print("\nUnique Filenames:")
        print(unique_filenames)

        num_files = len(unique_filenames)

        plt.figure(figsize=(10, 6))

        # Dynamically generate unique markers and colors based on the number of files
        marker_list = np.array(["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "X", "P"])
        color_list = plt.cm.get_cmap("tab20", num_files).colors if num_files > 10 else plt.cm.tab10.colors

        # Ensure we have enough unique markers and colors
        markers = {f: marker_list[i % len(marker_list)] for i, f in enumerate(unique_filenames)}
        colors = {f: color_list[i % len(color_list)] for i, f in enumerate(unique_filenames)}

        # Check if y-axis contains categorical values
        if data[y_axis].dtype == "object":
            unique_categories = sorted(data[y_axis].unique())
            category_map = {cat: i for i, cat in enumerate(unique_categories)}
            data[y_axis] = data[y_axis].map(category_map)

            category_legend_labels = [f"{num}: {cat}" for cat, num in category_map.items()]
        else:
            category_legend_labels = None

        # Plot each filename with a unique marker and color
        for filename, subset in data.groupby("filename"):
            # Use the full path for the legend if needed
            plt.plot(subset[x_axis], subset[y_axis],
                    marker=markers[filename], linestyle="-", color=colors[filename],
                    label=filename)  # `label=filename` is key here

        # Labels and title
        plt.xlabel(x_axis)
        plt.ylabel(f"{y_axis} (Normalized)" if normalize else y_axis)
        plt.title(f"{y_axis} vs {x_axis}")

        # Grid and main legend (filename-based)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Add categorical legend below the plot if applicable
        if category_legend_labels:
            plt.figtext(0.5, -0.1, "\n".join(category_legend_labels), ha="center", fontsize=10,
                        bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

        # Display plot in Streamlit
        st.pyplot(plt)




    def _render_right_column(self):
        """
        Renders the plotting area in the right column of the app.
        """
        st.header("Plotting Area")

        # Check if plot data is available
        if hasattr(self, 'plot_data_dict') and self.plot_data_dict is not None:
            try:
                # Call the _generate_plot method to render the plot
                self._generate_plot(
                    data=self.plot_data_dict['data'],
                    x_axis=self.plot_data_dict['x_axis'],
                    y_axis=self.plot_data_dict['y_axis'],
                    normalize=self.plot_data_dict.get('normalize', False)  # Default to False if not provided
                )
            except Exception as e:
                st.error(f"An error occurred while rendering the plot: {e}")
                logger.error(f"Plot rendering error: {e}", exc_info=True)
        else:
            st.write("No data to plot yet.")
        
 

if __name__ == "__main__":
    app = FileFilterApp(default_path="/Volumes/Titan/lotzegud/P08_test_data/healthy2")
    app.run()