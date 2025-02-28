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
    def __init__(self):
        self.path = "/Users/lotzegud/P08/fio_nxs_and_cmd_tool/"
        self.file_filter = ""
        self.extension_filter = ""
        self.selected_files = []
        self.plot_data = {}
        self.processed_data = {}
        self.nexus_processor = NeXusBatchProcessor(self.path)
        self.fio_processor = FioBatchProcessor(self.path)
    

    def run(self):
        st.title("NeXus-Fio-File Plotting App")

        # Create two columns, with the right column wider for plotting
        col1, col2 = st.columns([2, 3])

        with col1:
            self._render_left_column()

        with col2:
            self._render_right_column()
            
            

    def _render_left_column(self):
        st.header("File Selection")

        # Input field for directory path
        self.path = st.text_input(
            "Enter the directory path:", 
            value=self.path if self.path else "/Users/lotzegud/P08/fio_nxs_and_cmd_tool"
        )
        
        if st.button("Force Reload"):
            st.cache_data.clear()
            self.nexus_processor.process_files(force_reload=True)
            self.fio_processor.process_files(force_reload=True)
            self.processed_data.clear()
            st.rerun()

        if self.path and Path(self.path).is_dir():
            # Ensure session state keys exist
            if "extension_filter" not in st.session_state:
                st.session_state["extension_filter"] = "All"
            if "file_filter" not in st.session_state:
                st.session_state["file_filter"] = ""

            # File type selection (no duplicate widgets)
            st.radio("Select file type:", ["All", ".fio", ".nxs"], horizontal=True, key="extension_filter")

            # File name filter (triggers rerun automatically on change)
            st_keyup("Filter filenames by string:", key="file_filter", debounce=100) #debounce 100ms

            # Get filtered files dynamically
            filtered_files = self._get_filtered_files()

            # Display table with selectable rows
            self._render_selectable_table(filtered_files)

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
        min_width: int = 100, 
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




            
        
    def _render_selectable_table(self, filtered_files):
        """Displays a scrollable, multi-column table with selectable rows using st_aggrid."""
        if not filtered_files:
            st.write("No files match the filter.")
            return
        
    
        # Determine file types
        nxs_files = [f for f in filtered_files if f.endswith(".nxs")]
        fio_files = [f for f in filtered_files if f.endswith(".fio")]
        logger.debug(f"NXS files : {len(nxs_files)}")
        logger.debug(f"Fio files : {len(fio_files )}")
        logger.debug(f"Filtered files : {filtered_files}")

        # Fetch metadata for .nxs files
        nxs_metadata = None
        if nxs_files:
            nxs_metadata = self.nexus_processor.get_core_metadata().collect()

        # Fetch metadata for .fio files
        fio_metadata = None
        if fio_files:
            fio_metadata = self.fio_processor.get_core_metadata()
        
        # Combine metadata into a single DataFrame
        if nxs_metadata is not None and fio_metadata is not None:
            logger.debug(f"NXS Metadata Schema: {nxs_metadata.schema}")
            logger.debug(f"FIO Metadata Schema: {fio_metadata.schema}")
            logger.debug(f"Shortly before pl.concat")
            combined_metadata = pl.concat([nxs_metadata, fio_metadata])
        elif nxs_metadata is not None:
            combined_metadata = nxs_metadata
        elif fio_metadata is not None:
            combined_metadata = fio_metadata
        else:
            st.warning("No selection data available.")
            return
        
        # Debugging: Print lengths of filtered_files and combined_metadata
        logger.debug('nxs_metadata ', nxs_metadata)
        logger.debug('fio_metadata ', fio_metadata)
        
        logger.debug(f"Filtered files: {len(filtered_files)}")
        logger.debug(f"Combined metadata rows: {len(combined_metadata)}")

        

        # Debugging: Print filtered_files_for_metadata
        logger.debug(f"filtered_files: {filtered_files}")
        logger.debug(f"metadata_filenames: {combined_metadata['filename'].to_list()}")
        

        # Ensure combined_metadata contains only relevant filenames
        combined_metadata = combined_metadata.filter(
            pl.col("filename").is_in(filtered_files)
        )

        # Recompute last modified timestamps for the now-filtered files
        last_modified_values = [
            datetime.fromtimestamp(Path(self.path, f).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            for f in filtered_files if Path(self.path, f).exists()
        ]

        # Ensure length consistency
        if len(last_modified_values) != len(combined_metadata):
            raise ValueError(f"Mismatch: combined_metadata ({len(combined_metadata)}) vs last_modified_values ({len(last_modified_values)})")

        # Add "Last Modified" column safely
        combined_metadata = combined_metadata.with_columns([
            pl.Series("Last Modified", last_modified_values)
        ])


        # Sort by filename
        combined_metadata = combined_metadata.sort("filename")
        
        # **Compute optimal column widths before converting to Pandas**
        column_widths = self.compute_optimal_column_widths(combined_metadata)
        

        # Convert Polars DataFrame to Pandas (st_aggrid requires Pandas)
        df_pd = combined_metadata.to_pandas()
        
        logger.debug(f"Columns before AgGrid: {df_pd.columns.tolist()}")

        # Configure AgGrid options
        gb = GridOptionsBuilder.from_dataframe(df_pd)
        for col, width in column_widths.items():
            gb.configure_column(col, width=width, wrapText=True, autoHeight=True)
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)  # Enable checkboxes
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=10)  # Enable pagination
        grid_options = gb.build()
        grid_options["defaultColDef"] = {
                            "cellStyle": {"font-size": "10px"}  # Set the font size here
                        }

        # Display AgGrid table
        grid_response = AgGrid(df_pd, gridOptions=grid_options, height=400, fit_columns_on_grid_load=True)
        
        
        # Debugging: Log detailed grid response
        logger.debug("Grid Response Data: %s", grid_response.data)
        logger.debug("Grid Selected Rows: %s", grid_response.selected_rows)

        # Ensure grid_response is valid before proceeding
        # Ensure grid_response is valid before proceeding
        if not grid_response or not isinstance(grid_response.get("selected_rows"), list):
            st.warning("No selection data available.")
            self.selected_files = []
            return

        # Assign selected filenames, ensuring it's always a list
        #self.selected_files = [row["Filename"] for row in grid_response["selected_rows"] if "Filename" in row]

        # Update selected files
        self.selected_files = [row["Filename"] for row in grid_response["selected_rows"]]
                        
  
    def _get_filtered_files(self):
        """Returns a list of .fio or .nxs files in the directory that match the filters."""
        path = Path(self.path).resolve()  # Ensure absolute path
        if not path.is_dir():
            st.error(f"Invalid directory: {self.path}")
            return []

        # Read the extension and filename filters from session state
        ext_filter = st.session_state.get("extension_filter", "All")
        file_filter = st.session_state.get("file_filter", "").strip()  # Strip spaces

        valid_extensions = (".fio", ".nxs") if ext_filter == "All" else (ext_filter,)
        files = [f.name for f in path.iterdir() if f.suffix in valid_extensions]

        # Apply filename filtering (supports numbers)
        if file_filter:
            if file_filter.isdigit():  # If only numbers are entered, match scan number
                #regex_pattern = rf"\D{file_filter}\D"  # Ensures number isn't part of another number
                regex_pattern= rf"(?<!\d){file_filter}(?!\d)"
            else:
                regex_pattern = re.escape(file_filter).replace(r"\*", ".*").replace(r"\?", ".")
            
            files = [f for f in files if re.search(regex_pattern, f, re.IGNORECASE)]
            
        return files

    def _render_plot_options(self):
        st.header("Plot Options")

        if self.selected_files:
            # Example: Allow selecting time or motor vs signal
            x_axis = st.selectbox("Select X-axis:", ["Time", "Motor"])
            y_axis = st.selectbox("Select Y-axis:", ["Signal1", "Signal2"])
            normalize = st.checkbox("Normalize data")

            if st.button("Plot"):
                self._plot_data(x_axis, y_axis, normalize)

    def _plot_data(self, x_axis, y_axis, normalize):
        """Simulates plotting logic."""
        st.write(f"Plotting {x_axis} vs {y_axis} with normalization: {normalize}")
        # Here you would add your actual plotting logic using libraries like matplotlib or plotly
        # For now, we'll just display a placeholder
        st.write("Plot placeholder")

    def _render_right_column(self):
        st.header("Plotting Area")
        if self.plot_data:
            # Render the plot here
            pass
        else:
            st.write("No data to plot yet.")
            
 

if __name__ == "__main__":
    app = FileFilterApp()
    app.run()