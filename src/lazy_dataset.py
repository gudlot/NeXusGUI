# NeXusGUI – A GUI for visualising data across multiple NeXus files.
# Copyright (C) 2025 Gudrun Lotze
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
__copyright__ = "Copyright 2025"
__license__ = "AGPL-3.0"
__status__ = "Development"

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
