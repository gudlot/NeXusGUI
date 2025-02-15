import streamlit as st
import polars as pl
import h5py
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

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
        
        # Calculate dynamic height based on the number of rows
        # Set a base height of 100 pixels and add 35 pixels per row (with a small buffer)
        dynamic_height = max(50, 35 * df.height + 35)  # Added 20 pixels as a buffer
            
        # Display the DataFrame with enhanced features
        st.dataframe(
            df.to_pandas(),  # Convert to Pandas for better Streamlit integration
            use_container_width=True,
            height=dynamic_height,  # Dynamically adjust height based on the number of rows
            column_config={
                col: st.column_config.Column(
                    help=f"Column: {col}",
                    width="medium"
                )
                for col in df.columns
            }
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