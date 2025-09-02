import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

LoaderType = Literal["csv", "tsv", "html", "txt", "json"]


class DataLoader:
    """
    Handles loading data from various file formats into a pandas DataFrame.

    The loader type is automatically inferred from the file extension.
    Supported formats: csv, tsv, html, txt.

    The class can be initialized with or without a default data path. If no path
    is provided during initialization, it must be provided when calling the
    `load_data` method.

    Attributes:
        data_path (Optional[str]): The path to the data file. Can be set or
                                   changed later.
    """

    def __init__(self, data_path: Optional[str] = None, encoding: str = "utf-8"):
        """
        Initializes the DataLoader.

        Args:
            data_path (Optional[str], optional): The default path to the data file.
                                                 Defaults to None.
            encoding (str, optional): The file encoding to use. Defaults to "utf-8".
        """
        self.data_path = data_path
        self._encoding = encoding

        self._load_method_map: Dict[LoaderType, Callable[..., pd.DataFrame]] = {
            "csv": self._load_csv,
            "tsv": self._load_tsv,
            "html": self._load_html,
            "txt": self._load_txt,
            "json": self._load_json,
        }

        if data_path:
            logger.info(f"DataLoader initialized for default path='{self.data_path}'")
        else:
            logger.info("DataLoader initialized without a default data path.")

    def load_data(self, path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Loads data from a file into a pandas DataFrame.

        The method uses the path provided here, or the default path from initialization
        if no new path is given. One of them must be present.

        Args:
            path (Optional[str], optional): The path to the file to load. If provided,
                                            it overrides the default path for this call.
                                            Defaults to None.
            **kwargs: Additional keyword arguments to be passed directly to the
                      underlying pandas read function (e.g., `sep`, `header`, `names`).

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            ValueError: If no data path is available (neither in `__init__` nor in `path`).
            FileNotFoundError: If the file at the specified path does not exist.
        """
        current_path = path or self.data_path
        if not current_path:
            raise ValueError(
                "No data path provided. Please supply a path either during "
                "initialization or when calling load_data."
            )

        try:
            loader_type = self._get_loader_type_from_path(current_path)
            load_method = self._load_method_map[loader_type]

            logger.info(
                f"Attempting to load data from '{current_path}' using method '{loader_type}'"
            )
            return load_method(current_path, **kwargs)

        except FileNotFoundError:
            logger.error(f"File not found at path: {current_path}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to load data from {current_path}: {e}",
                exc_info=True,
            )
            raise

    def _get_loader_type_from_path(self, path_str: str) -> LoaderType:
        """Determines the loader type from the file extension using pathlib."""
        path_obj = Path(path_str)
        suffix = path_obj.suffix.lower()
        if suffix == ".csv":
            return "csv"
        elif suffix == ".tsv":
            return "tsv"
        elif suffix in [".html", ".htm"]:
            return "html"
        elif suffix == ".txt":
            return "txt"
        elif suffix == ".json":
            return "json"
        else:
            raise ValueError(f"Could not determine loader type for file: {path_str}")

    def _load_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """Implements the logic for loading a CSV file."""
        logger.debug(f"Executing _load_csv on {path} with kwargs: {kwargs}")
        return pd.read_csv(path, encoding=self._encoding, **kwargs)

    def _load_tsv(self, path: str, **kwargs) -> pd.DataFrame:
        """Implements the logic for loading a TSV file."""
        logger.debug(f"Executing _load_tsv on {path} with kwargs: {kwargs}")
        # Set default separator to tab, but allow user to override it via kwargs
        kwargs.setdefault("sep", "\t")
        return pd.read_csv(path, encoding=self._encoding, **kwargs)

    def _load_html(self, path: str, **kwargs) -> pd.DataFrame:
        """Implements the logic for loading the first table from an HTML file."""
        logger.debug(f"Executing _load_html on {path} with kwargs: {kwargs}")
        # `flavor` is a good candidate for default, user can override
        kwargs.setdefault("flavor", "lxml")
        tables = pd.read_html(path, encoding=self._encoding, **kwargs)
        if not tables:
            raise ValueError(f"No tables found in HTML file: {path}")
        logger.info(f"Found {len(tables)} tables in HTML, returning the first one.")
        return tables[0]

    def _load_txt(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Implements the logic for loading a TXT file (one entry per line).
        This method reads the file line by line and creates a single-column DataFrame.
        """
        logger.debug(f"Executing _load_txt on {path}")
        with open(path, "r", encoding=self._encoding) as f:
            lines = [line.strip() for line in f.readlines()]

        col_name = kwargs.get("names", ["text"])[0]

        return pd.DataFrame(lines, columns=[col_name])

    def _load_json(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Implements the logic for loading a JSON file into a DataFrame.
        Converts JSON data to DataFrame using pd.json_normalize().
        """
        logger.debug(f"Executing _load_json on {path}")
        try:
            with open(path, "r", encoding=self._encoding) as f:
                data = json.load(f)

            # המר ל-DataFrame
            if isinstance(data, list):
                # אם זה רשימה של אובייקטים
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                # אם זה אובייקט אחד
                df = pd.json_normalize([data])
            else:
                # אם זה ערך פשוט
                df = pd.DataFrame([data])

            logger.info(f"Loaded JSON with shape: {df.shape}")
            return df

        except FileNotFoundError:
            logger.error(f"File not found at path: {path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load JSON from {path}: {e}")
            raise

    def _load_mapping(self, path: str) -> Dict[str, Any]:
        """
        Loads Elasticsearch mapping from JSON file.
        Returns the raw dictionary (not a DataFrame).
        """
        logger.debug(f"Executing _load_mapping on {path}")
        try:
            with open(path, "r", encoding=self._encoding) as f:
                mapping = json.load(f)

            logger.info(f"Successfully loaded mapping from {path}")
            return mapping

        except FileNotFoundError:
            logger.error(f"Mapping file not found at path: {path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in mapping file {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load mapping from {path}: {e}")
            raise

    def load_mapping(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads an Elasticsearch mapping from a JSON file.

        Args:
            path (Optional[str], optional): The path to the mapping file. If provided,
                                          it overrides the default path for this call.
                                          Defaults to None.

        Returns:
            Dict[str, Any]: The mapping dictionary.

        Raises:
            ValueError: If no data path is available.
            FileNotFoundError: If the file at the specified path does not exist.
        """
        current_path = path or self.data_path
        if not current_path:
            raise ValueError(
                "No data path provided. Please supply a path either during "
                "initialization or when calling load_mapping."
            )

        return self._load_mapping(current_path)

    def load_lines_as_list(
        self, path: Optional[str] = None, strip_empty: bool = True
    ) -> List[str]:
        """Load text file lines directly as a list of strings."""
        current_path = path or self.data_path
        if not current_path:
            raise ValueError("No data path provided.")

        try:
            with open(current_path, "r", encoding=self._encoding) as f:
                lines = [line.strip() for line in f.readlines()]

            if strip_empty:
                lines = [line for line in lines if line]

            logger.info(f"Loaded {len(lines)} lines from {current_path}")
            return lines

        except Exception as e:
            logger.error(f"Failed to load lines from {current_path}: {e}")
            raise
