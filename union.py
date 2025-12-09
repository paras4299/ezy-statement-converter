#Working - Union Bank of India Statement Extractor
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
class UnionExtractorConfig:
    """Configuration for Union Bank Statement Extractor."""
    # Columns expected in the transaction table
    transaction_columns: List[str] = field(default_factory=lambda: [
        'Tran Id', 'Tran Date', 'Remarks', 'Amount (Rs.)', 'Balance (Rs.)'
    ])
    
    # Columns for merging continuation rows
    merge_cols: List[str] = field(default_factory=lambda: [
        'Remarks', 'Amount (Rs.)', 'Balance (Rs.)'
    ])

class UnionStatementExtractor:
    """
    Extracts transaction tables from Union Bank statement PDFs using pdfplumber.
    Supports password-protected PDFs.
    """

    def __init__(self, pdf_file: str, password: str = '', config: Optional[UnionExtractorConfig] = None):
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
        self.config = config or UnionExtractorConfig()
        logger.info(f"Initialized Union Bank StatementExtractor for {self.pdf_file}")

    def extract_tables(self) -> pd.DataFrame:
        """
        Orchestrates the extraction process across all pages.
        """
        logger.info(f"Starting extraction for {self.pdf_file}")
        
        all_rows = []
        
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
                    
                    # Extract tables using pdfplumber
                    tables = page.extract_tables()
                    
                    if not tables:
                        logger.warning(f"No tables found on page {page_num}.")
                        continue
                    
                    for table_idx, table in enumerate(tables):
                        logger.info(f"Processing table {table_idx + 1} on page {page_num}")
                        
                        # Convert to DataFrame
                        if not table or len(table) == 0:
                            continue
                        
                        df = pd.DataFrame(table)
                        
                        # Check if this is a transaction table
                        if self._is_transaction_table(df):
                            cleaned_df = self._clean_table(df, page_num)
                            if not cleaned_df.empty:
                                all_rows.append(cleaned_df)

        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            raise

        if not all_rows:
            logger.warning("No transaction data extracted from any page.")
            return pd.DataFrame()

        # Concatenate all pages
        final_df = pd.concat(all_rows, ignore_index=True)
        
        return final_df

    def _is_transaction_table(self, df: pd.DataFrame) -> bool:
        """
        Check if the DataFrame is a transaction table.
        """
        if df.empty or df.shape[1] < 3:
            return False
        
        # Check if first few rows contain transaction-related keywords
        first_rows_text = ' '.join(df.iloc[:3].astype(str).values.flatten())
        keywords = ['Tran Id', 'Tran Date', 'Remarks', 'Amount', 'Balance']
        
        return any(keyword.lower() in first_rows_text.lower() for keyword in keywords)

    def _clean_table(self, df: pd.DataFrame, page_num: int) -> pd.DataFrame:
        """
        Clean the extracted table.
        """
        # Remove header rows
        header_mask = df.iloc[:, 0].astype(str).str.contains(
            'Tran Id|Tran Date|Statement|DETAILS OF STATEMENT', 
            case=False, 
            na=False, 
            regex=True
        )
        df = df[~header_mask]
        df = df.reset_index(drop=True)
        
        # Remove empty rows
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.replace('None', pd.NA)
        df = df.dropna(how='all', axis=0)
        df = df.reset_index(drop=True)
        
        if df.empty:
            return pd.DataFrame()
        
        # Assign column names based on number of columns
        num_cols = df.shape[1]
        
        if num_cols == 5:
            df.columns = self.config.transaction_columns
        elif num_cols == 4:
            # Missing Tran Id
            logger.info(f"Page {page_num}: Handling 4 columns (missing Tran Id)")
            df.columns = ['Tran Date', 'Remarks', 'Amount (Rs.)', 'Balance (Rs.)']
            df.insert(0, 'Tran Id', pd.NA)
            df = df[self.config.transaction_columns]
        elif num_cols == 3:
            logger.info(f"Page {page_num}: Handling 3 columns")
            df.columns = ['Remarks', 'Amount (Rs.)', 'Balance (Rs.)']
            df.insert(0, 'Tran Id', pd.NA)
            df.insert(1, 'Tran Date', pd.NA)
            df = df[self.config.transaction_columns]
        else:
            logger.warning(f"Page {page_num}: Unexpected number of columns ({num_cols}). Skipping.")
            return pd.DataFrame()
        
        return df

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the transactions DataFrame (merges rows, fixes dates, converts numbers).
        """
        logger.info("Cleaning transactions data...")
        
        if df.empty:
            return df
        
        # Merge Continuation Rows
        rows_to_drop = []
        last_valid_idx = -1
        
        for col in self.config.merge_cols:
            if col not in df.columns:
                df[col] = ""
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Check if Tran Date is empty (continuation)
            tran_date_val = str(row['Tran Date']).strip() if pd.notna(row['Tran Date']) else ""
            
            is_continuation = (tran_date_val == "")
            
            if is_continuation and last_valid_idx != -1:
                # Merge
                for col in self.config.merge_cols:
                    val = str(row[col]).strip() if pd.notna(row[col]) else ""
                    if val:
                        current_val = str(df.at[last_valid_idx, col]).strip() if pd.notna(df.at[last_valid_idx, col]) else ""
                        if current_val:
                            df.at[last_valid_idx, col] = current_val + " " + val
                        else:
                            df.at[last_valid_idx, col] = val
                rows_to_drop.append(i)
            else:
                if tran_date_val:
                    last_valid_idx = i
                elif last_valid_idx == -1:
                    rows_to_drop.append(i)

        if rows_to_drop:
            logger.info(f"Merging {len(rows_to_drop)} continuation rows.")
            df = df.drop(rows_to_drop).reset_index(drop=True)

        # Fix Dates - Union Bank uses format like "17/02/2022"
        for col in ['Tran Date']:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str).replace('nan', '').replace('<NA>', '')
            
            # Clean up the date strings
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # Try multiple date formats
            # First try: "17/02/2022" format
            parsed_dates = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
            
            # Second try: "17-02-2022" format for rows that failed
            mask = parsed_dates.isna() & (df[col] != '')
            if mask.any():
                parsed_dates.loc[mask] = pd.to_datetime(df.loc[mask, col], format='%d-%m-%Y', errors='coerce')
            
            # Third try: "17 Feb 2022" format
            mask = parsed_dates.isna() & (df[col] != '')
            if mask.any():
                parsed_dates.loc[mask] = pd.to_datetime(df.loc[mask, col], format='%d %b %Y', errors='coerce')
            
            df[col] = parsed_dates.dt.date

        # Parse Amount and Balance - Union Bank format: "300.00 (Dr)" or "10000.00 (Cr)"
        # Split into Debit and Credit columns
        if 'Amount (Rs.)' in df.columns:
            # Extract amount and type
            df['Amount_Clean'] = df['Amount (Rs.)'].astype(str).str.extract(r'([\d,\.]+)')[0]
            df['Amount_Type'] = df['Amount (Rs.)'].astype(str).str.extract(r'\((Dr|Cr)\)', flags=re.IGNORECASE)[0]
            
            # Convert to numeric
            df['Amount_Clean'] = df['Amount_Clean'].str.replace(',', '', regex=False)
            df['Amount_Clean'] = pd.to_numeric(df['Amount_Clean'], errors='coerce')
            
            # Create Debit and Credit columns
            df['Debit'] = df.apply(lambda row: row['Amount_Clean'] if str(row['Amount_Type']).upper() == 'DR' else 0, axis=1)
            df['Credit'] = df.apply(lambda row: row['Amount_Clean'] if str(row['Amount_Type']).upper() == 'CR' else 0, axis=1)
            
            # Drop temporary columns
            df = df.drop(['Amount_Clean', 'Amount_Type'], axis=1)
        
        # Clean Balance column
        if 'Balance (Rs.)' in df.columns:
            df['Balance (Rs.)'] = df['Balance (Rs.)'].astype(str).str.replace(',', '', regex=False)
            df['Balance (Rs.)'] = df['Balance (Rs.)'].str.extract(r'([\d\.]+)')[0]
            df['Balance (Rs.)'] = pd.to_numeric(df['Balance (Rs.)'], errors='coerce')
        
        # Fill missing values with 0 for Debit and Credit
        for col in ['Debit', 'Credit']:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Reorder columns
        cols = ['Tran Id', 'Tran Date', 'Remarks', 'Debit', 'Credit', 'Balance (Rs.)']
        df = df[[c for c in cols if c in df.columns]]

        return df

    def save_to_excel(self, df: pd.DataFrame, output_file: str):
        """
        Saves the processed DataFrame to an Excel file with formatting.
        """
        logger.info(f"Saving to {output_file}...")
        try:
            with pd.ExcelWriter(output_file, engine='xlsxwriter', date_format='dd/mm/yyyy') as writer:
                df.to_excel(writer, sheet_name='Transactions', index=False)
                
                # Auto-adjust columns
                worksheet = writer.sheets['Transactions']
                for idx, col in enumerate(df.columns):
                    max_len = max(
                        df[col].astype(str).map(len).max(),
                        len(str(col))
                    ) + 2
                    if col == 'Remarks':
                        max_len = 50
                    worksheet.set_column(idx, idx, max_len)
                    
            logger.info("Done.")
        except Exception as e:
            logger.error(f"Error saving Excel: {e}")

if __name__ == "__main__":
    # Example usage
    pdf_file = r'.\Statements\UNION.pdf'
    output_excel = r'.\Statements\UNION.xlsx'
    
    password = input("Enter PDF password (press Enter if not password-protected): ").strip()
    
    try:
        extractor = UnionStatementExtractor(pdf_file, password=password)
        df = extractor.extract_tables()
        
        if not df.empty:
            logger.info("Preview Transactions (Raw):")
            print(df.head(10))
            
            df = extractor.clean_dataframe(df)
            
            logger.info("Preview Transactions (Cleaned):")
            print(df.head(10))
            
            extractor.save_to_excel(df, output_excel)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
