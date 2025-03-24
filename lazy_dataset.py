from dataclasses import dataclass
from pathlib import Path
import h5py


@dataclass
class LazyDatasetReference:
    directory: Path
    file_name: str
    dataset_name: str

    def load_on_demand(self):
        """Load dataset from HDF5 file lazily."""
        file_path = self.directory / self.file_name
        with h5py.File(file_path, "r") as f:
            data = f[self.dataset_name][:]
            
            # Check for empty data (0 elements)
            if data.size == 0:
                return None  # Missing or empty data
            
            if data.ndim == 1:
                if data.size == 0:
                    return None  # Empty 1D array should return None
                return data
                
            elif data.ndim == 2:
                # Handle empty 2D arrays with zero elements
                if data.shape[0] == 0 or data.shape[1] == 0:
                    return None  # 2D array with 0 rows or 0 columns is treated as None
                return data 

            else:
                raise ValueError(f"Unexpected shape {data.shape} in dataset {self.dataset_name}")
