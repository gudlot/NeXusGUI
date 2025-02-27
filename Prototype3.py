import streamlit as st
from st_keyup import st_keyup
from pathlib import Path
import os
import polars as pl
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import logging
import re
import h5py 




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

            
        
    def _render_selectable_table(self, filtered_files):
        """Displays a scrollable, multi-column table with selectable rows using st_aggrid."""
        if not filtered_files:
            st.write("No files match the filter.")
            return
        # Find the longest filename length
        max_filename_length = max(len(f) for f in filtered_files) if filtered_files else 60

        # Approximate width based on character count (10 pixels per character)
        estimated_width = max(600, min(10 * max_filename_length, 800))  # Min 600px, Max 800px


        # Create a Polars DataFrame with additional metadata
        df = pl.DataFrame({
            "Filename": filtered_files,
            "Size (KB)": [round(Path(self.path, f).stat().st_size / 1024, 2) for f in filtered_files],
            "Last Modified": [Path(self.path, f).stat().st_mtime for f in filtered_files]
        }).sort("Filename")

        # Convert Polars DataFrame to Pandas (st_aggrid requires Pandas)
        df_pd = df.to_pandas()

        # Configure AgGrid options
        gb = GridOptionsBuilder.from_dataframe(df_pd)
        # Set column width for Filename to be large enough
        gb.configure_column("Filename", width=estimated_width, wrapText=True, autoHeight=True)
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
        if grid_response is None or "selected_rows" not in grid_response:
            st.warning("No selection data available.")
            self.selected_files = []
            return

        # Ensure selected_rows is always a list before processing
        selected_rows = grid_response.get("selected_rows", []) # Always a list (default empty list)
            
  
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
                regex_pattern = rf"\D{file_filter}\D"  # Ensures number isn't part of another number
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