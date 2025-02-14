import streamlit as st
import polars as pl
import h5py
import numpy as np
from pathlib import Path

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


def main():
    st.title("NXS File Browser")

    # Set the default path
    default_path = "/Users/lotzegud/P08/fio_nxs_and_cmd_tool"
    selected_path = st.text_input("Enter directory path:", default_path)

    # List .nxs files in the selected directory
    if st.button("List .nxs files"):
        folder_path = Path(selected_path)
        if folder_path.is_dir():
            nxs_files = sorted([f for f in folder_path.glob("*.nxs")])
            if not nxs_files:
                st.write("No .nxs files found in the selected directory.")
            else:
                st.session_state.nxs_files = nxs_files
        else:
            st.error("Invalid directory. Please enter a valid path.")

    # If .nxs files are listed, allow selection
    if "nxs_files" in st.session_state:
        # Add a "Select All" checkbox
        select_all = st.checkbox("Select All")

        # Determine the default selection based on the "Select All" checkbox
        default_selection = [f.name for f in st.session_state.nxs_files] if select_all else []

        # Display the multiselect widget with the updated selection
        selected_files = st.multiselect(
            "Select Nexus files:",
            [f.name for f in st.session_state.nxs_files],
            default=default_selection
        )
        
        # Debug: Print selected files
        st.write("Selected Files:", selected_files)
        
        if selected_files:
            # Convert selected file names to full paths
            full_paths = [Path(selected_path) / file for file in selected_files]
            
            # Debug: Print full paths
            st.write("Full Paths:", full_paths)
            
            # Process the selected files and display the results
            processor = NexusDataProcessor()
            df = processor.process_multiple_files(full_paths)
            if not df.is_empty():
                st.write("### Data Overview")
                st.dataframe(df)
            else:
                st.write("No data found in the selected files.")
        else:
            st.write("No files selected.")


if __name__ == "__main__":
    main()