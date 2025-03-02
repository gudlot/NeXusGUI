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
        # Use session state if available; otherwise, use the default
        if "current_path" not in st.session_state:
            st.session_state["current_path"] = default_path

        self.path = st.session_state["current_path"]
        self.file_filter = ""
        self.extension_filter = ""
        
        logger.debug(f"Current directory path: {self.path}")

        
        self.selected_files = []
        self.selected_metadata = None
        self.plot_data = {}
        self.processed_data = {}
        
        self.nexus_processor = NeXusBatchProcessor(self.path)
        self.fio_processor = FioBatchProcessor(self.path)
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

    def run(self):
        st.title("NeXus-Fio-File Plotting App")

        # Create two columns, with the right column wider for plotting
        col1, col2 = st.columns([2, 3])

        with col1:
            self._render_left_column()

        with col2:
            self._render_right_column()
            
    def _initialize_processors(self):
        """Reinitialize processors with the current path."""
        self.nexus_processor = NeXusBatchProcessor(self.path)
        self.fio_processor = FioBatchProcessor(self.path)

    def _render_left_column(self):
        st.header("File Selection")

        # Get user input for the directory path
        new_path = st.text_input("Enter the directory path:", value=st.session_state["current_path"])

        # Update path only if it actually changed
        if new_path and new_path != st.session_state["current_path"]:
            if Path(new_path).is_dir():
                logger.info(f"Updating path to: {new_path}")

                # Update session state and class variable
                st.session_state["current_path"] = new_path
                self.path = new_path

                # Reset selected files and metadata
                self.selected_files = []
                self.selected_metadata = None
                st.session_state["selected_files"] = []
                st.session_state["selected_metadata"] = None

                # Clear cache and reinitialize processors
                st.cache_data.clear()
                self._initialize_processors()

                # Rerun to refresh UI
                st.rerun()
            else:
                st.error(f"Invalid directory: {new_path}")
                                    
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
            combined_metadata = pl.concat([nxs_metadata, fio_metadata])
        elif nxs_metadata is not None:
            combined_metadata = nxs_metadata
        elif fio_metadata is not None:
            combined_metadata = fio_metadata
        else:
            st.warning("No selection data available.")
            return
        
        # Debugging: Print lengths of filtered_files and combined_metadata
        #logger.debug('nxs_metadata ', nxs_metadata)
        #logger.debug('fio_metadata ', fio_metadata)
        #logger.debug(f"Filtered files: {len(filtered_files)}")
        #logger.debug(f"Combined metadata rows: {len(combined_metadata)}")

        

        # Debugging: Print filtered_files_for_metadata
        logger.debug(f"filtered_files: {filtered_files}")
        logger.debug(f"metadata_filenames: {combined_metadata['filename'].to_list()}")
        

        # Ensure combined_metadata contains only relevant filenames
        combined_metadata = combined_metadata.filter(
            pl.col("filename").is_in(filtered_files)
        )

        # Recompute creation timestamps for the now-filtered files
        created_values = [
            datetime.fromtimestamp(Path(self.path, f).stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            for f in filtered_files if Path(self.path, f).exists()
        ]

        # Ensure length consistency
        if len(created_values) != len(combined_metadata):
            raise ValueError(f"Mismatch: combined_metadata ({len(combined_metadata)}) vs created_values ({len(created_values)})")

        ## Add "Created" column safely
        #combined_metadata = combined_metadata.with_columns([
        #    pl.Series("Created", created_values)
        #])


        # Sort by filename
        #combined_metadata = combined_metadata.sort("filename")
        #Sort by scan_id
        combined_metadata = combined_metadata.with_columns(
        pl.col("scan_id").cast(pl.Int64, strict=False)  # Convert to integer if possible
    ).sort("scan_id")
        
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
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=20)  # Enable pagination
        grid_options = gb.build()
        grid_options["defaultColDef"] = {
                            "cellStyle": {"font-size": "10px"}  # Set the font size here
                        }

        # Display AgGrid table
        grid_response = AgGrid(df_pd, gridOptions=grid_options, height=400, fit_columns_on_grid_load=True)
        
        
        # Debugging: Log detailed grid response
        logger.debug("Grid Response Data: %s", grid_response.data)
        logger.debug("Grid Selected Rows: %s", grid_response.selected_rows)
        
        # Debugging: Log detailed grid response before validation
        logger.debug(f"Grid Response Keys: {list(grid_response.keys()) if grid_response else 'None'}")
        logger.debug(f"Grid Response Data: {grid_response}")
        logger.debug(f"Selected Rows Content: {grid_response['selected_rows']}")
        logger.debug(f'type {type(grid_response['selected_rows'])}')

        # Extract selected filenames from the AgGrid response
        if grid_response and "selected_rows" in grid_response:
            selected_df = pd.DataFrame(grid_response["selected_rows"])
            if "filename" in selected_df.columns:
                st.session_state["selected_files"] = selected_df["filename"].tolist()
                self.selected_files = st.session_state["selected_files"] 
                
                # Store selected metadata as a Polars DataFrame
                self.selected_metadata = pl.from_pandas(selected_df)
                st.session_state["selected_metadata"] = self.selected_metadata
            else:
                self.selected_metadata = None
                st.session_state["selected_metadata"] = None  # Clear if nothing selected

        # Debugging: Verify selected files
        logger.debug("\N{hot pepper}")
        logger.debug(f"Selected files: {self.selected_files}")

    def _get_filtered_files(self):
        """Returns a list of .fio or .nxs files in the directory that match the filters."""
        
        
        path = Path(self.path).resolve()  # Ensure absolute path
        if not path.is_dir():
            st.error(f"Invalid directory: {self.path}")
            return []
        
        logger.debug(f"Current directory path: {self.path}")

        # Read the extension and filename filters from session state
        ext_filter = st.session_state.get("extension_filter", "All")
        file_filter = st.session_state.get("file_filter", "").strip()  # Strip spaces

        valid_extensions = (".fio", ".nxs") if ext_filter == "All" else (ext_filter,)
        # List files in the directory (non-recursive)
        #files = [f.name for f in path.iterdir() if f.is_file() and f.suffix in valid_extensions]
        files = [f.name for f in path.glob("*") if f.is_file() and f.suffix in valid_extensions]

        
        
        # Log the files found in the directory
        logger.debug(f"Files found in directory: {files}")

        # Apply filename filtering (supports numbers)
        if file_filter:
            if file_filter.isdigit():  # If only numbers are entered, match scan number
                #regex_pattern = rf"\D{file_filter}\D"  # Ensures number isn't part of another number
                regex_pattern= rf"(?<!\d){file_filter}(?!\d)"
            else:
                regex_pattern = re.escape(file_filter).replace(r"\*", ".*").replace(r"\?", ".")
            
            files = [f for f in files if re.search(regex_pattern, f, re.IGNORECASE)]
            
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

                self._plot_data(x_axis, y_axis, z_axis, normalize)




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