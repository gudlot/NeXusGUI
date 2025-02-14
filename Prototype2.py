import streamlit as st
import polars as pl
import h5py
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

# Abstract Base Class (ABC) for GUI
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

    @abstractmethod
    def get_selected_files(self):
        pass


# Streamlit Implementation of the GUI Interface
class StreamlitGUI(GUI):
    def render(self, browser):
        st.title("NXS File Browser")

        # Set the default path
        browser.selected_path = st.text_input("Enter directory path:", browser.root_directory)

        # List .nxs files in the selected directory
        if st.button("List .nxs files"):
            browser.list_nxs_files()
        
        # Debug: Print the list of .nxs files
        if hasattr(browser, "nxs_files"):
            st.write("List of .nxs files:", [f.name for f in browser.nxs_files])

        if hasattr(browser, "nxs_files") and browser.nxs_files:
            # Add a "Select All" checkbox
            select_all = st.checkbox("Select All")

            # Determine the default selection based on the "Select All" checkbox
            default_selection = [f.name for f in browser.nxs_files] if select_all else []

            # Display the multiselect widget with the updated selection
            selected_files = st.multiselect(
                "Select Nexus files:",
                [f.name for f in browser.nxs_files],
                default=default_selection
            )
            
            # Debug: Print selected files
            st.write("Selected Files:", selected_files)
            
            if selected_files:
                # Convert selected file names to full paths
                full_paths = [Path(browser.selected_path) / file for file in selected_files]
                
                # Debug: Print full paths
                st.write("Full Paths:", full_paths)
                
                # Process the selected files and display the results
                df = browser.processor.process_multiple_files(full_paths)
                if not df.is_empty():
                    st.write("### Data Overview")
                    st.dataframe(df)
                else:
                    st.write("No data found in the selected files.")
            else:
                st.write("No files selected.")

    def show_message(self, message):
        st.write(message)

    def show_error(self, error):
        st.error(error)

    def get_selected_files(self):
        return st.session_state.get("selected_files", [])


# NexusDataProcessor: Handles data extraction and processing
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
                # Recursively extract data from groups
                self.extract_data(item, full_path + "/", data_dict)
            elif isinstance(item, h5py.Dataset):
                try:
                    data = item[()]  # Read dataset contents
                    if isinstance(data, np.ndarray):
                        # Convert numpy arrays to lists for easier handling
                        data_dict[full_path] = data.tolist()
                    elif isinstance(data, (bytes, str)):
                        # Decode bytes to strings
                        data_dict[full_path] = data.decode("utf-8") if isinstance(data, bytes) else data
                    else:
                        # Handle other types (e.g., numbers)
                        data_dict[full_path] = str(data)
                except Exception as e:
                    data_dict[full_path] = f"Error: {e}"
        return data_dict

    def process_single_file(self, file_path: Path) -> dict:
        with h5py.File(file_path, "r") as f:
            if "scan" in f:
                # Extract data from the "scan" group
                data_dict = self.extract_data(f["scan"], "/scan/")
                data_dict["filename"] = file_path.name
                return data_dict
            else:
                return {"filename": file_path.name}

    def process_multiple_files(self, file_paths: list) -> pl.DataFrame:
        all_data = [self.process_single_file(file_path) for file_path in file_paths]
        self.data = pl.DataFrame(all_data, schema_overrides={k: pl.Object for k in all_data[0] if k != "filename"})
        return self.data


# NXSFileBrowser: Core logic for file browsing and processing
class NXSFileBrowser:
    # Add this to the NXSFileBrowser class
    def __init__(self, gui: GUI):
        if "browser" not in st.session_state:
            st.session_state.browser = self
        self.root_directory = "/Users/lotzegud/P08/fio_nxs_and_cmd_tool"
        self.selected_path = self.root_directory
        self.nxs_files = st.session_state.get("nxs_files", [])
        self.processor = NexusDataProcessor()
        self.gui = gui

    # Modify the list_nxs_files method
    def list_nxs_files(self):
        folder_path = Path(self.selected_path)
        if folder_path.is_dir():
            self.nxs_files = sorted([f for f in folder_path.glob("*.nxs")])
            st.session_state.nxs_files = self.nxs_files  # Persist in session state
            if not self.nxs_files:
                self.gui.show_message("No .nxs files found in the selected directory.")
        else:
            self.gui.show_error("Invalid directory. Please enter a valid path.")

    def run(self):
        self.gui.render(self)


# Main Application
if __name__ == "__main__":
    # Initialize the Streamlit GUI
    gui = StreamlitGUI()

    # Create and run the file browser
    browser = NXSFileBrowser(gui)
    browser.run()