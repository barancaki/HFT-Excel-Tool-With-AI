import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os
from Levenshtein import ratio
import tempfile
from concurrent.futures import ThreadPoolExecutor
import gc

class DataMatcher:
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold / 100  # Convert percentage to decimal

    def _read_excel_file(self, file_path: str) -> pd.DataFrame:
        """Read Excel file and return DataFrame."""
        try:
            df = pd.read_excel(file_path)
            # Convert all column names to strings and clean them
            df.columns = df.columns.astype(str).str.strip()
            return df
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            raise

    def _clean_value(self, value) -> str:
        """Clean and standardize value for comparison."""
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _find_matching_columns(self, df1: pd.DataFrame, df2: pd.DataFrame, target_columns: List[str]) -> List[Tuple[str, str]]:
        """Find matching columns from the target columns list."""
        matching_columns = []
        
        # Clean and standardize target column names
        target_columns = [col.strip() for col in target_columns]
        
        print("\nLooking for specified columns:")
        for target in target_columns:
            print(f"\nSearching for column: '{target}'")
            
            # Look for exact or similar matches in both DataFrames
            matches_df1 = [col for col in df1.columns if self._is_column_match(target, col)]
            matches_df2 = [col for col in df2.columns if self._is_column_match(target, col)]
            
            if matches_df1 and matches_df2:
                print(f"Found matching columns for '{target}':")
                print(f"- File 1: {matches_df1}")
                print(f"- File 2: {matches_df2}")
                
                # Add all possible combinations of matching columns
                for col1 in matches_df1:
                    for col2 in matches_df2:
                        matching_columns.append((col1, col2))
            else:
                print(f"No matching columns found for '{target}'")
        
        return matching_columns

    def _is_column_match(self, target: str, column: str) -> bool:
        """Check if a column matches the target column name."""
        target = target.strip().lower()
        column = column.strip().lower()
        
        # Check exact match
        if target == column:
            return True
        
        # Check similarity
        similarity = ratio(target, column)
        return similarity >= self.similarity_threshold

    def _find_matching_rows(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                          matching_columns: List[Tuple[str, str]]) -> pd.DataFrame:
        """Find matching rows based on specified columns."""
        if not matching_columns:
            return pd.DataFrame()

        print("\nFinding matching rows...")
        matches = []
        
        # Group matching columns
        column_groups = {}
        for col1, col2 in matching_columns:
            if col1 not in column_groups:
                column_groups[col1] = []
            column_groups[col1].append(col2)

        # Process each column group
        for col1, cols2 in column_groups.items():
            print(f"\nProcessing matches for column '{col1}'")
            
            for col2 in cols2:
                try:
                    print(f"Comparing with column '{col2}'")
                    
                    # Clean values for comparison and filter out empty/whitespace values
                    df1_clean = df1.copy()
                    df2_clean = df2.copy()
                    df1_clean[col1] = df1_clean[col1].apply(self._clean_value)
                    df2_clean[col2] = df2_clean[col2].apply(self._clean_value)
                    
                    # Filter out rows where the matching columns contain only whitespace or empty values
                    df1_clean = df1_clean[df1_clean[col1].str.strip() != '']
                    df2_clean = df2_clean[df2_clean[col2].str.strip() != '']
                    
                    if df1_clean.empty or df2_clean.empty:
                        print(f"No valid data to compare after filtering empty values in columns {col1} and {col2}")
                        continue
                    
                    # Find exact matches
                    merged = pd.merge(
                        df1_clean,
                        df2_clean,
                        left_on=col1,
                        right_on=col2,
                        suffixes=('_file1', '_file2'),
                        how='inner'
                    )
                    
                    if not merged.empty:
                        merged['Match_Type'] = 'Exact Match'
                        merged['Matched_Column_File1'] = col1
                        merged['Matched_Column_File2'] = col2
                        matches.append(merged)
                        print(f"Found {len(merged)} exact matches")
                    
                    # Find fuzzy matches
                    if self.similarity_threshold < 1.0:  # Only if not requiring exact matches
                        left_only = df1_clean[~df1_clean[col1].isin(df2_clean[col2])]
                        right_only = df2_clean[~df2_clean[col2].isin(df1_clean[col1])]
                        
                        fuzzy_matches = []
                        for idx1, row1 in left_only.iterrows():
                            val1 = self._clean_value(row1[col1])
                            for idx2, row2 in right_only.iterrows():
                                val2 = self._clean_value(row2[col2])
                                if val1 and val2:  # Skip empty values
                                    if ratio(val1, val2) >= self.similarity_threshold:
                                        match_row = pd.concat([row1, row2])
                                        match_row['Match_Type'] = 'Fuzzy Match'
                                        match_row['Matched_Column_File1'] = col1
                                        match_row['Matched_Column_File2'] = col2
                                        match_row['Similarity'] = ratio(val1, val2)
                                        fuzzy_matches.append(match_row)
                        
                        if fuzzy_matches:
                            fuzzy_df = pd.DataFrame(fuzzy_matches)
                            matches.append(fuzzy_df)
                            print(f"Found {len(fuzzy_matches)} fuzzy matches")
                    
                except Exception as e:
                    print(f"Error matching rows for columns {col1} and {col2}: {str(e)}")
                    continue

        if matches:
            # Combine all matches and remove duplicates
            result = pd.concat(matches, ignore_index=True)
            result = result.drop_duplicates()
            print(f"\nTotal matches found: {len(result)}")
            return result
        
        print("No matching rows found")
        return pd.DataFrame()

    def _prepare_dataframe_for_excel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for Excel export."""
        if df.empty:
            return pd.DataFrame({'Message': ['No data available']})

        try:
            # Create a copy to avoid modifying the original
            df_clean = df.copy()

            # Convert all columns to string type to avoid Excel compatibility issues
            for col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str)

            # Replace NaN, None, and empty strings with a placeholder
            df_clean = df_clean.replace({np.nan: '', None: '', 'nan': '', 'None': ''})

            return df_clean
        except Exception as e:
            print(f"Error preparing DataFrame: {str(e)}")
            return pd.DataFrame({'Error': [f'Error preparing data: {str(e)}']})

    def find_matches(self, file_paths: List[str], target_columns: List[str]) -> str:
        """Find matching data across multiple Excel files for specific columns."""
        if len(file_paths) < 2:
            raise ValueError("At least two files are required for matching")
        
        if not target_columns:
            raise ValueError("At least one target column must be specified")

        print(f"Processing {len(file_paths)} files for matching data...")
        print(f"Target columns: {', '.join(target_columns)}")
        
        # Create a temporary file for results
        result_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.xlsx',
            prefix='matching_data_'
        ).name

        try:
            # Create Excel writer with xlsxwriter engine
            with pd.ExcelWriter(result_file, engine='xlsxwriter') as writer:
                summary_data = []
                
                # Compare each pair of files
                for i, file1 in enumerate(file_paths):
                    for j, file2 in enumerate(file_paths[i+1:], i+1):
                        try:
                            print(f"\nComparing {os.path.basename(file1)} with {os.path.basename(file2)}")
                            
                            # Read files
                            df1 = self._read_excel_file(file1)
                            df2 = self._read_excel_file(file2)
                            
                            print(f"File 1 columns: {', '.join(df1.columns)}")
                            print(f"File 2 columns: {', '.join(df2.columns)}")
                            
                            # Find matching columns from target columns
                            matching_columns = self._find_matching_columns(df1, df2, target_columns)
                            
                            if matching_columns:
                                print(f"\nFound {len(matching_columns)} matching column pairs:")
                                for col1, col2 in matching_columns:
                                    print(f"- '{col1}' matches '{col2}'")
                                
                                # Find matching rows
                                matches = self._find_matching_rows(df1, df2, matching_columns)
                                
                                if not matches.empty:
                                    # Create sheet name
                                    sheet_name = f"Matches_{i+1}_{j+1}"
                                    
                                    # Prepare and write matches
                                    matches_clean = self._prepare_dataframe_for_excel(matches)
                                    matches_clean.to_excel(writer, sheet_name=sheet_name, index=False)
                                    
                                    # Add to summary
                                    summary_data.append({
                                        'File 1': os.path.basename(file1),
                                        'File 2': os.path.basename(file2),
                                        'Matching Columns': len(matching_columns),
                                        'Matching Rows': len(matches),
                                        'Sheet Name': sheet_name
                                    })
                                    
                                    print(f"Found {len(matches)} matching rows")
                                else:
                                    print("No matching rows found")
                            else:
                                print("No matching columns found")
                                
                        except Exception as e:
                            print(f"Error processing files {file1} and {file2}: {str(e)}")
                            continue
                        
                        # Clear memory
                        gc.collect()
                
                # Write summary sheet
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df = self._prepare_dataframe_for_excel(summary_df)
                else:
                    summary_df = pd.DataFrame({'Message': ['No matches found between any files']})
                
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            return result_file

        except Exception as e:
            print(f"Error in Excel writing process: {str(e)}")
            if os.path.exists(result_file):
                try:
                    os.remove(result_file)
                except:
                    pass
            raise Exception(f"Error generating Excel file: {str(e)}") 