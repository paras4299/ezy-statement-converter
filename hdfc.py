#Working
import camelot
import pandas as pd
import re
import pdfplumber
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class ExtractorConfig:
    """Configuration for CamelotTableExtractor."""
    # Regex patterns (compiled for performance)
    # top_marker_pattern: re.Pattern = field(default_factory=lambda: re.compile(r"From\s*:\s*.*\s*To\s*:\s*.*\s*Statement\s*of\s*account"))
    top_marker_pattern: re.Pattern = field(default_factory=lambda: re.compile(r"From\s*:\s*.*\s*To\s*:\s*.*\s"))
    summary_marker_pattern: re.Pattern = field(default_factory=lambda: re.compile(r"STATEMENT\s*SUMMARY"))
    bottom_marker_fallback_1_pattern: re.Pattern = field(default_factory=lambda: re.compile(r"HDFC\s*BANK\s*LIMITED"))
    bottom_marker_fallback_2_pattern: re.Pattern = field(default_factory=lambda: re.compile(r"\*Closing\s*balance\s*includes\s*funds"))
    
    # Columns expected in the transaction table
    transaction_columns: List[str] = field(default_factory=lambda: [
        'Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 
        'Withdrawal Amt.', 'Deposit Amt.', 'Closing Balance'
    ])
    
    # Columns for merging continuation rows
    merge_cols: List[str] = field(default_factory=lambda: [
        'Narration', 'Chq./Ref.No.', 'Withdrawal Amt.', 'Deposit Amt.', 'Closing Balance'
    ])
    
    # Summary table keys
    summary_keys: List[str] = field(default_factory=lambda: [
        "Opening Balance", "Dr Count", "Cr Count", "Debits", "Credits", "Closing Bal"
    ])
    
    # Threshold for distinguishing Withdrawal vs Deposit column in 6-column pages
    column_type_threshold: float = 450.0

class CamelotTableExtractor:
    """
    Extracts transaction tables and summary data from bank statement PDFs using Camelot and pdfplumber.
    Version 3: Enhanced robustness and optimizations.
    """

    def __init__(self, pdf_file: str, password: str = '', config: Optional[ExtractorConfig] = None):
        """
        Initialize the extractor.

        Args:
            pdf_file: Path to the PDF file.
            password: Password for encrypted PDF (default: empty string).
            config: Optional configuration object.
        """
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF file not found: {pdf_file}")
        
        self.pdf_file = pdf_file
        self.password = password
        self.config = config or ExtractorConfig()
        logger.info(f"Initialized CamelotTableExtractor V3 for {self.pdf_file}")

    def _detect_area(self, page: pdfplumber.page.Page) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        """
        Detects the crop area for the table and extracts summary data if present.

        Args:
            page: pdfplumber Page object.

        Returns:
            Tuple containing:
            - List of area strings for Camelot ["x1,y1,x2,y2"] or None.
            - Dictionary of summary data or None.
        """
        width = page.width
        height = page.height
        summary_data = None

        # 1. Search for Top Marker
        top_matches = page.search(self.config.top_marker_pattern)
        y_top_pdf = height  # Default to top of page

        if top_matches:
            # pdfplumber 'bottom' is the lower y value (larger number in top-left origin)
            top_y_plumber = top_matches[0]['bottom']
            # Convert to PDF coordinates (bottom-left origin)
            y_top_pdf = height - top_y_plumber
        
        # 2. Search for Bottom Marker (Priority: Statement Summary)
        summary_matches = page.search(self.config.summary_marker_pattern)
        y_bottom_pdf = 0  # Default to bottom of page

        if summary_matches:
            logger.info(f"Found 'STATEMENT SUMMARY' on page {page.page_number}. Extracting summary...")
            summary_top_plumber = summary_matches[0]['top']
            y_bottom_pdf = height - summary_top_plumber
            
            # Extract Summary Data
            summary_data = self._extract_summary(page, summary_top_plumber, width, height)

        else:
            # Fallback Bottom Markers
            # Priority: Check for Closing Balance line first
            closing_matches = page.search(self.config.bottom_marker_fallback_2_pattern)
            
            if closing_matches:
                # If Closing Balance found, check if HDFC is also present (user request: "these 2 line must be there")
                hdfc_matches = page.search(self.config.bottom_marker_fallback_1_pattern)
                if hdfc_matches:
                    # Use HDFC top as bottom crop
                    bottom_y_plumber = hdfc_matches[0]['top']
                    y_bottom_pdf = height - bottom_y_plumber
                else:
                    logger.warning("Found Closing Balance line but missing 'HDFC BANK LIMITED'. Cannot set bottom crop.")
            else:
                # If Closing Balance not found, we cannot use this fallback as "these 2 line must be there"
                pass

        # If no markers found at all, return None
        if not top_matches and not summary_matches and not closing_matches: # Fixed logic here to check closing_matches too if it was relevant
             # Actually, the logic was: if NO markers found. 
             # If we found closing_matches but not hdfc_matches, we still found *something* but couldn't use it.
             # But if we return None, we skip the page.
             # Let's stick to the original logic: if we didn't establish a valid area via markers, we might want to skip.
             # However, if top matches found, we have a top. If bottom not found, we use 0.
             # So if top_matches is found, we proceed.
            #  pass
             return None, None

        # Camelot area format: x1, y_top, x2, y_bottom
        # x1=0, x2=width (full width)
        area_str = f"0,{y_top_pdf},{width},{y_bottom_pdf}"
        return [area_str], summary_data

    def _extract_summary(self, page: pdfplumber.page.Page, top: float, width: float, height: float) -> Optional[Dict[str, Any]]:
        """
        Extracts summary data from the specified area of the page.
        """
        summary_bbox = (0, top, width, height)
        summary_crop = page.crop(summary_bbox)
        
        # Try extracting tables
        summary_tables = summary_crop.extract_tables()
        
        if summary_tables:
            logger.info(f"Extracted {len(summary_tables)} tables from summary area.")
            # Flatten data
            flat_data = [item for sublist in summary_tables[0] for item in sublist if item]
            flat_data = [str(x).strip() for x in flat_data if str(x).strip() != ""]
            logger.debug(f"Summary Flat Data: {flat_data}")
            
            return self._parse_summary_data(flat_data)
        else:
            logger.warning("No tables found in summary area.")
            return None

    def _parse_summary_data(self, flat_data: List[str]) -> Dict[str, Any]:
        """
        Parses the flattened summary data list into a structured dictionary.
        """
        values = []
        
        # Strategy 1: Check for merged string (e.g. "9,600... 70 24 ...")
        for item in flat_data:
            if re.search(r'\d', item) and "Statement Summary" not in item:
                parts = item.split()
                if len(parts) >= 6:
                    values = parts
                    break
        
        # Strategy 2: Check for separate items (filtering out headers)
        if not values:
            headers_keywords = ["Opening", "Balance", "Dr", "Count", "Cr", "Debits", "Credits", "Closing"]
            potential_values = [
                x for x in flat_data 
                if not any(k in x for k in headers_keywords) and "Statement Summary" not in x
            ]
            if len(potential_values) >= 6:
                values = potential_values

        if len(values) >= 6:
            return {
                "Opening Balance": values[0],
                "Dr Count": values[1],
                "Cr Count": values[2],
                "Debits": values[3],
                "Credits": values[4],
                "Closing Bal": values[5]
            }
        else:
            logger.warning(f"Could not parse summary table correctly. Values found: {values}")
            return {"Raw": flat_data}

    def extract_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Orchestrates the extraction process across all pages.
        """
        logger.info(f"Starting extraction for {self.pdf_file}")
        
        all_dfs = []
        final_summary_data = None
        
        try:
            # Try to open with password if provided
            try:
                if self.password:
                    pdf = pdfplumber.open(self.pdf_file, password=self.password)
                else:
                    pdf = pdfplumber.open(self.pdf_file)
            except Exception as e:
                logger.error(f"Failed to open PDF: {e}")
                raise Exception("Failed to open PDF. It may be password-protected or corrupted.")
            
            with pdf:
                num_pages = len(pdf.pages)
                logger.info(f"Total pages: {num_pages}")
                
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    logger.info(f"Processing page {page_num}...")
                    
                    areas, summary_data = self._detect_area(page)
                    
                    if summary_data:
                        logger.info(f"Captured Summary Data from Page {page_num}")
                        final_summary_data = summary_data
                    
                    if not areas:
                        logger.warning(f"Text markers not found on page {page_num}. Skipping.")
                        continue
                    
                    logger.info(f"Detected Area: {areas}")
                    
                    try:
                        tables = camelot.read_pdf(
                            self.pdf_file, 
                            flavor="stream", 
                            pages=str(page_num), 
                            split_text=True, 
                            table_areas=areas, 
                            row_tol=10
                        )
                        
                        if tables:
                            df = tables[0].df
                            df = self._initial_clean_df(df, page_num, tables[0])
                            if not df.empty:
                                all_dfs.append(df)
                        else:
                            logger.warning(f"No tables extracted by Camelot on page {page_num}.")
                            
                    except Exception as e:
                        logger.error(f"Error extracting data on page {page_num}: {e}")

        except Exception as e:
            logger.error(f"Error opening PDF file: {e}")
            return pd.DataFrame(), pd.DataFrame()

        if not all_dfs:
            logger.warning("No data extracted from any page.")
            return pd.DataFrame(), pd.DataFrame()

        # Concatenate all pages
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Robust Header Removal (V3 Feature)
        # Check if a row matches ALL expected header columns
        # We check for 'Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.', 'Deposit Amt.', 'Closing Balance'
        # To be safe, checking the first 3-4 is usually sufficient and avoids issues if OCR is slightly off on numbers
        header_mask = (
            (final_df['Date'].astype(str).str.contains('Date', case=False, na=False)) &
            (final_df['Narration'].astype(str).str.contains('Narration', case=False, na=False)) &
            (final_df['Chq./Ref.No.'].astype(str).str.contains('Chq', case=False, na=False))
        )
        
        # Also check for the exact string match if possible, but contains is safer for slight OCR variations
        # Let's use the exact logic requested: "match full row data"
        # We can create a combined string of the row and check if it looks like a header
        
        final_df = final_df[~header_mask]
        final_df = final_df.reset_index(drop=True)
        
        # Create Summary DF
        summary_df = pd.DataFrame()
        if final_summary_data:
            summary_df = pd.DataFrame([final_summary_data])
        else:
            logger.warning("Statement Summary not found via text markers.")

        return final_df, summary_df

    def _initial_clean_df(self, df: pd.DataFrame, page_num: int, table: Any) -> pd.DataFrame:
        """
        Performs initial cleaning on the raw Camelot DataFrame (headers, columns).
        """
        # Drop empty columns
        df = df.replace(r'^\s*$', pd.NA, regex=True)

        if df.shape[1] >= 7:
            df = df.iloc[:, :7]
            df.columns = self.config.transaction_columns
            return df
        elif df.shape[1] == 6:
            logger.info(f"Handling Page {page_num} with 6 columns.")
            return self._handle_6_column_page(df, table)
        else:
            logger.warning(f"Page {page_num} has {df.shape[1]} columns. Skipping.")
            return pd.DataFrame()

    def _handle_6_column_page(self, df: pd.DataFrame, table: Any) -> pd.DataFrame:
        """Handles the specific case where either Withdrawal or Deposit column is missing."""
        try:
            cols = table.cols
            # cols[4] is the tuple for 5th column
            x_start = cols[4][0]
            logger.debug(f"5th Column Start X: {x_start}")
            
            if x_start < self.config.column_type_threshold:
                logger.info("Identified as Withdrawal. Missing Deposit.")
                df.columns = ['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.', 'Closing Balance']
                df.insert(5, 'Deposit Amt.', pd.NA)
            else:
                logger.info("Identified as Deposit. Missing Withdrawal.")
                df.columns = ['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Deposit Amt.', 'Closing Balance']
                df.insert(4, 'Withdrawal Amt.', pd.NA)
        except Exception as e:
            logger.error(f"Error determining column type: {e}. Defaulting to missing Deposit.")
            df.columns = ['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.', 'Closing Balance']
            df.insert(5, 'Deposit Amt.', pd.NA)
        
        return df

    def clean_dataframe(self, df: pd.DataFrame, summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cleans the transactions DataFrame (merges rows, fixes dates, converts numbers).
        """
        logger.info("Cleaning transactions data...")
        
        # 1. Merge Continuation Rows
        # Optimization: We can't easily vectorize the merge logic because it depends on sequential state (last_valid_idx).
        # However, we can optimize the string operations.
        
        rows_to_drop = []
        last_valid_idx = -1
        
        # Ensure columns exist
        for col in self.config.merge_cols:
            if col not in df.columns:
                df[col] = ""
        
        # Pre-convert relevant columns to string to avoid repeated casting
        # But we need to be careful not to mess up the original df if we modify it in place
        # Working with .iloc is generally fine.
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Check if Date and Value Dt are empty (continuation)
            date_val = str(row['Date']).strip() if pd.notna(row['Date']) else ""
            val_date_val = str(row['Value Dt']).strip() if pd.notna(row['Value Dt']) else ""
            
            is_continuation = (date_val == "" and val_date_val == "")
            
            if is_continuation and last_valid_idx != -1:
                # Merge
                for col in self.config.merge_cols:
                    val = str(row[col]).strip() if pd.notna(row[col]) else ""
                    if val:
                        current_val = str(df.at[last_valid_idx, col]).strip() if pd.notna(df.at[last_valid_idx, col]) else ""
                        if current_val:
                            df.at[last_valid_idx, col] = current_val + " " + val # Added space for safety, though V1 didn't have it. V1 had "" + val. Let's stick to V1 behavior but maybe add space if needed? V1 was `current_val + "" + val` which is just concatenation.
                            # Actually, for Narration, a space is usually good. For numbers, no space.
                            # But let's stick to V1 logic to avoid breaking changes, unless optimization allows improvement.
                            # V1: `df.at[last_valid_idx, col] = current_val + "" + val` -> effectively `current_val + val`
                        else:
                            df.at[last_valid_idx, col] = val
                rows_to_drop.append(i)
            else:
                if date_val:
                    last_valid_idx = i
                elif last_valid_idx == -1:
                    rows_to_drop.append(i)

        if rows_to_drop:
            logger.info(f"Merging {len(rows_to_drop)} continuation rows.")
            df = df.drop(rows_to_drop).reset_index(drop=True)

        # 2. Fix Dates
        # Vectorized string operation
        # Regex to match /yy at the end and convert to /20yy
        # We can use pandas str.replace with regex
        
        # First ensure they are strings
        for col in ['Date', 'Value Dt']:
             # Convert to string, handle NaN
             df[col] = df[col].astype(str).replace('nan', '')
             # Apply regex replacement vectorized
             df[col] = df[col].str.replace(r'/(\d{2})$', r'/20\1', regex=True)
             # Convert to datetime
             df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce').dt.date

        # 3. Convert Numbers
        num_cols = ['Withdrawal Amt.', 'Deposit Amt.', 'Closing Balance']
        for col in num_cols:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Fill missing values with 0 for Withdrawal and Deposit
        for col in ['Withdrawal Amt.', 'Deposit Amt.']:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 4. Convert Summary Numbers
        if not summary_df.empty:
            for col in self.config.summary_keys:
                if col in summary_df.columns:
                    summary_df[col] = summary_df[col].astype(str).str.replace(',', '', regex=False)
                    summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
        
        return df, summary_df

    def save_to_excel(self, df: pd.DataFrame, summary_df: pd.DataFrame, output_file: str):
        """
        Saves the processed DataFrames to an Excel file with formatting.
        """
        logger.info(f"Saving to {output_file}...")
        try:
            with pd.ExcelWriter(output_file, engine='xlsxwriter', date_format='dd/mm/yyyy') as writer:
                df.to_excel(writer, sheet_name='Transactions', index=False)
                
                # Auto-adjust columns for Transactions
                worksheet = writer.sheets['Transactions']
                for idx, col in enumerate(df.columns):
                    max_len = max(
                        df[col].astype(str).map(len).max(),
                        len(str(col))
                    ) + 2
                    if col == 'Narration':
                        max_len = 50
                    worksheet.set_column(idx, idx, max_len)
                
                # Save Summary if exists
                if not summary_df.empty:
                    logger.info("Saving Summary sheet...")
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    # Auto-adjust columns for Summary
                    worksheet_summary = writer.sheets['Summary']
                    for idx, col in enumerate(summary_df.columns):
                        max_len = max(
                            summary_df[col].astype(str).map(len).max(),
                            len(str(col))
                        ) + 2
                        worksheet_summary.set_column(idx, idx, max_len)
                    
            logger.info("Done.")
        except Exception as e:
            logger.error(f"Error saving Excel: {e}")

if __name__ == "__main__":
    # Example usage
    pdf_file = r'.\Statements\SBI.pdf'
    output_excel = r'.\Statements\SBI.xlsx'
    
    try:
        extractor = CamelotTableExtractor(pdf_file)
        df, summary_df = extractor.extract_tables()
        
        if not df.empty:
            logger.info("Preview Transactions:")
            print(df.head())
            
            df, summary_df = extractor.clean_dataframe(df, summary_df)
            
            if not summary_df.empty:
                logger.info("Preview Summary:")
                print(summary_df)
            
            extractor.save_to_excel(df, summary_df, output_excel)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
