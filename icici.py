#Working - ICICI Bank Statement Extractor
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
class ICICIExtractorConfig:
    """Configuration for ICICI Bank Statement Extractor."""
    transaction_columns: List[str] = field(default_factory=lambda: [
        'DATE', 'MODE', 'PARTICULARS', 'DEPOSITS', 'WITHDRAWALS', 'BALANCE'
    ])
    
    merge_cols: List[str] = field(default_factory=lambda: [
        'PARTICULARS', 'DEPOSITS', 'WITHDRAWALS', 'BALANCE'
    ])

class ICICIStatementExtractor:
    """
    Extracts transaction tables from ICICI Bank statement PDFs using pdfplumber.
    """

    def __init__(self, pdf_file: str, password: str = '', config: Optional[ICICIExtractorConfig] = None):
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF file not found: {pdf_file}")
        
        self.pdf_file = pdf_file
        self.password = password
        self.config = config or ICICIExtractorConfig()
        logger.info(f"Initialized ICICI Bank StatementExtractor for {self.pdf_file}")

    def extract_tables(self) -> pd.DataFrame:
        logger.info(f"Starting extraction for {self.pdf_file}")
        
        all_rows = []
        
        try:
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
                    
                    # Extract text and parse transactions
                    text = page.extract_text()
                    if text:
                        transactions = self._parse_text_transactions(text, page_num)
                        if transactions:
                            all_rows.extend(transactions)

        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            raise

        if not all_rows:
            logger.warning("No transaction data extracted from any page.")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows, columns=self.config.transaction_columns)
        return df
    
    def _parse_text_transactions(self, text: str, page_num: int) -> List[List]:
        """Parse transactions from text."""
        transactions = []
        lines = text.split('\n')
        
        # Skip header lines
        start_idx = 0
        for i, line in enumerate(lines):
            if 'DATE MODE PARTICULARS DEPOSITS WITHDRAWALS BALANCE' in line:
                start_idx = i + 1
                break
        
        i = start_idx
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if line starts with a date pattern (DD-MM-YYYY)
            date_match = re.match(r'^(\d{2}-\d{2}-\d{4})\s*(.*)', line)
            if date_match:
                date = date_match.group(1)
                rest_of_line = date_match.group(2).strip()
                
                # Collect all lines until next date or end
                particulars_lines = [rest_of_line] if rest_of_line else []
                j = i + 1
                
                while j < len(lines):
                    next_line = lines[j].strip()
                    # Stop if we hit another date
                    if re.match(r'^\d{2}-\d{2}-\d{4}', next_line):
                        break
                    # Stop if empty line or page footer
                    if not next_line or 'Page' in next_line or 'ICICI Bank' in next_line:
                        j += 1
                        continue
                    particulars_lines.append(next_line)
                    j += 1
                
                # Join all particulars
                full_text = ' '.join(particulars_lines)
                
                # Extract all numbers (amounts) from the text
                # Look for patterns like "300.00" or "4,285.91"
                amounts = re.findall(r'([\d,]+\.\d{2})', full_text)
                
                mode = ''
                particulars = full_text
                deposits = ''
                withdrawals = ''
                balance = ''
                
                # Logic: Last number is always balance
                # If 2 numbers: withdrawal + balance OR deposit + balance
                # If 3 numbers: deposit + withdrawal + balance
                if len(amounts) >= 1:
                    balance = amounts[-1]
                    # Remove balance from particulars
                    particulars = full_text.replace(balance, '', 1).strip()
                    
                    if len(amounts) == 2:
                        # Could be deposit or withdrawal
                        # Check if it's before balance in text
                        first_amount = amounts[0]
                        # Remove from particulars
                        particulars = particulars.replace(first_amount, '', 1).strip()
                        
                        # Heuristic: if particulars contains "CMS TRANSACTION" it's usually a deposit
                        if 'CMS TRANSACTION' in particulars or 'CREDIT' in particulars.upper():
                            deposits = first_amount
                        else:
                            withdrawals = first_amount
                    
                    elif len(amounts) >= 3:
                        # First is deposit, second is withdrawal (or vice versa)
                        # Check the order in text
                        first_amount = amounts[0]
                        second_amount = amounts[1]
                        
                        # Remove from particulars
                        particulars = particulars.replace(first_amount, '', 1).strip()
                        particulars = particulars.replace(second_amount, '', 1).strip()
                        
                        # Usually: deposit comes first, then withdrawal
                        deposits = first_amount
                        withdrawals = second_amount
                
                # Clean up particulars - remove extra spaces
                particulars = re.sub(r'\s+', ' ', particulars).strip()
                
                transactions.append([date, mode, particulars, deposits, withdrawals, balance])
                i = j
            else:
                i += 1
        
        logger.info(f"Extracted {len(transactions)} transactions from page {page_num}")
        return transactions

    def _is_transaction_table(self, df: pd.DataFrame) -> bool:
        if df.empty or df.shape[1] < 4:
            return False
        
        first_rows_text = ' '.join(df.iloc[:3].astype(str).values.flatten())
        keywords = ['DATE', 'MODE', 'PARTICULARS', 'DEPOSITS', 'WITHDRAWALS', 'BALANCE']
        
        return any(keyword.lower() in first_rows_text.lower() for keyword in keywords)

    def _clean_table(self, df: pd.DataFrame, page_num: int) -> pd.DataFrame:
        # Remove header rows
        header_mask = df.iloc[:, 0].astype(str).str.contains(
            'DATE|MODE|PARTICULARS|Statement|ACCOUNT DETAILS', 
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
        
        num_cols = df.shape[1]
        
        if num_cols == 6:
            df.columns = self.config.transaction_columns
        elif num_cols == 5:
            logger.info(f"Page {page_num}: Handling 5 columns")
            df.columns = ['DATE', 'PARTICULARS', 'DEPOSITS', 'WITHDRAWALS', 'BALANCE']
            df.insert(1, 'MODE', pd.NA)
            df = df[self.config.transaction_columns]
        elif num_cols == 4:
            logger.info(f"Page {page_num}: Handling 4 columns")
            df.columns = ['DATE', 'PARTICULARS', 'WITHDRAWALS', 'BALANCE']
            df.insert(1, 'MODE', pd.NA)
            df.insert(3, 'DEPOSITS', pd.NA)
            df = df[self.config.transaction_columns]
        else:
            logger.warning(f"Page {page_num}: Unexpected number of columns ({num_cols}). Skipping.")
            return pd.DataFrame()
        
        return df

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
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
            
            date_val = str(row['DATE']).strip() if pd.notna(row['DATE']) else ""
            
            is_continuation = (date_val == "" or date_val == "B/F")
            
            if is_continuation and last_valid_idx != -1:
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
                if date_val and date_val != "B/F":
                    last_valid_idx = i
                elif last_valid_idx == -1:
                    rows_to_drop.append(i)

        if rows_to_drop:
            logger.info(f"Merging {len(rows_to_drop)} continuation rows.")
            df = df.drop(rows_to_drop).reset_index(drop=True)

        # Fix Dates - ICICI uses format like "02-06-2025"
        for col in ['DATE']:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str).replace('nan', '').replace('<NA>', '')
            
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # Try multiple date formats
            parsed_dates = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
            
            mask = parsed_dates.isna() & (df[col] != '')
            if mask.any():
                parsed_dates.loc[mask] = pd.to_datetime(df.loc[mask, col], format='%d/%m/%Y', errors='coerce')
            
            mask = parsed_dates.isna() & (df[col] != '')
            if mask.any():
                parsed_dates.loc[mask] = pd.to_datetime(df.loc[mask, col], format='%d %b %Y', errors='coerce')
            
            df[col] = parsed_dates.dt.date

        # Convert Numbers
        num_cols = ['DEPOSITS', 'WITHDRAWALS', 'BALANCE']
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with 0
        for col in ['DEPOSITS', 'WITHDRAWALS']:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Rename columns to standard format
        df = df.rename(columns={
            'DATE': 'Date',
            'MODE': 'Mode',
            'PARTICULARS': 'Particulars',
            'DEPOSITS': 'Credit',
            'WITHDRAWALS': 'Debit',
            'BALANCE': 'Balance'
        })

        return df

    def save_to_excel(self, df: pd.DataFrame, output_file: str):
        logger.info(f"Saving to {output_file}...")
        try:
            with pd.ExcelWriter(output_file, engine='xlsxwriter', date_format='dd/mm/yyyy') as writer:
                df.to_excel(writer, sheet_name='Transactions', index=False)
                
                worksheet = writer.sheets['Transactions']
                for idx, col in enumerate(df.columns):
                    max_len = max(
                        df[col].astype(str).map(len).max(),
                        len(str(col))
                    ) + 2
                    if col == 'Particulars':
                        max_len = 50
                    worksheet.set_column(idx, idx, max_len)
                    
            logger.info("Done.")
        except Exception as e:
            logger.error(f"Error saving Excel: {e}")

if __name__ == "__main__":
    pdf_file = r'.\Statements\ICICI.pdf'
    output_excel = r'.\Statements\ICICI.xlsx'
    
    password = input("Enter PDF password (press Enter if not password-protected): ").strip()
    
    try:
        extractor = ICICIStatementExtractor(pdf_file, password=password)
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
