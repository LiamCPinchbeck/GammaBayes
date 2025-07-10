import pandas as pd
import numpy as np

class CSVDictionary:
    """
    A class to read a CSV file into a dictionary-like object where each column is accessible as a key.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the CSV data.
        dict_object (dict): The dictionary representation of the DataFrame.
    """
    def __init__(self, filepath, delimiter=' ', numeric_type=np.float64):
        """
        Initializes the CSVDictionary object by reading a CSV file and converting it to a dictionary.

        Args:
            filepath (str): The path to the CSV file.
            delimiter (str, optional): The delimiter used in the CSV file. Defaults to ' '.
            numeric_type (type, optional): The numeric type to which numeric columns should be converted. Defaults to np.float128.
        """
        self.df = pd.read_csv(filepath, delimiter=delimiter)
        self._convert_numeric_columns(numeric_type)

        self._create_dict()



    def _convert_numeric_columns(self, numeric_type):
        """
        Converts all columns that can be converted to the specified numeric type.

        Args:
            numeric_type (type): The numeric type to which columns should be converted.
        """
        for column in self.df.columns:
            # Attempt to convert each column to the specified numeric type.
            # If a column contains non-numeric data, it will remain unchanged.
            self.df[column] = pd.to_numeric(self.df[column], errors='ignore', downcast='float')
            if self.df[column].dtype == np.number or self.df[column].dtype == float:
                self.df[column] = self.df[column].astype(numeric_type)
        

    def _create_dict(self):
        """
        Converts the DataFrame to a dictionary where each key is a column header
        and each value is a NumPy array of column values.
        """
        self.dict_object = self.df.to_dict('list')

        try:
            del self.dict_object['Unnamed: 0']
        except:
            pass

        for key in self.dict_object:
            self.dict_object[key] = np.asarray(self.dict_object[key])

    def __delitem__(self, key):
        """
        Deletes an item using dictionary-like syntax.

        Args:
            key (str): The key to delete.

        Raises:
            KeyError: If the key is not found in the dictionary.
        """
        if key in self.dict_object:
            del self.dict_object[key]
        else:
            raise KeyError(f"Key '{key}' not found in the dictionary.")


    def __getitem__(self, key):
        """
        Allows column-based access like obj['column_name'] to get all values under that column.

        Args:
            key (str): The key to access.

        Returns:
            np.ndarray: The values under the specified column.
        """
        return self.dict_object[key]
        
    def __setitem__(self, key, values):
        """
        Allows setting or updating column values.

        Args:
            key (str): The key to set or update.
            values (np.ndarray): The values to set for the specified key.
        """
        # This simplistic implementation assumes 'values' is a complete list of column values.
        # You might need more complex handling depending on your exact requirements.
        self.dict_object[key] = values

        
    def keys(self):
        """
        Returns a list of column headers.

        Returns:
            list: The list of column headers.
        """
        return self.dict_object.keys()
    
    def values(self):
        """
        Returns a list of NumPy arrays with the values for each column.

        Returns:
            list: The list of NumPy arrays.
        """
        return self.dict_object.values()
    
    def items(self):
        """
        Returns a list of tuples, each tuple being (column_header, column_values_as_numpy_array).

        Returns:
            list: The list of tuples.
        """
        return self.dict_object.items()
    

    def pop(self, key, default=None):
        """
        Pops a column from the DataFrame, returning the column values.

        Args:
            key (str): The key to pop.
            default: The default value to return if the key is not found. Defaults to None.

        Returns:
            np.ndarray: The values under the specified column, or the default value if the key is not found.

        Raises:
            KeyError: If the key is not found and no default is provided.
        """
        if key in self.dict_object:
            return self.dict_object[key]
        else:
            if default is not None:
                return default
            else:
                raise KeyError(f"Column '{key}' does not exist.")


