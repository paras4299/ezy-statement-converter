# Kotak Bank Statement Extractor
import pandas as pd
import re
import pdfplumber
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class KotakExtractorConfig:
    """Configuration for Kotak Bank Statement Extractor."""
    transaction_columns: List[str] = field(default_factory=lambda: [
        'DATE', 'TRANSACTION DETAILS', 'CHEQUE/REFERENCE#', 'DEBIT', 'CREDIT', 'BALANCE'
    ])
    
    merge_cols: List[str] = field(default_factory=lambda: [
        'TRANSACTION DETAILS', 'CHEQUE/REFERENCE#', 'DEBIT', 'CREDIT', 'BALANCE'
    ])

class KotakStatementExtractor:
    """
    Extracts transaction tables from Kotak Bank statement PDFs using pdfplumber.
    """

    def __init__(self, pdf_file: str, password: str = '', config: Optional[KotakExtractorConfig] = None):
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF file not found: {pdf_file}")
        
        self.pdf_file = pdf_file
        self.password = password
        self.config = config or KotakExtractorConfig()
        logger.info(f"Initialized Kotak Bank StatementExtractor for {self.pdf_file}")

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
        
        # Skip header lines - look for the table header
        start_idx = 0
        for i, line in enumerate(lines):
            if 'DATE TRANSACTION DETAILS CHEQUE/REFERENCE#' in line or 'DATE TRANSACTION DETAILS' in line:
                start_idx = i + 1
                break
        
        i = start_idx
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if line starts with a date pattern (DD MMM, YYYY)
            date_match = re.match(r'^(\d{2}\s+[A-Za-z]{3},\s+\d{4})', line)
            if date_match:
                date = date_match.group(1)
                rest_of_line = line[len(date):].strip()
                
                # Collect all lines until next date or end
                transaction_lines = [rest_of_line] if rest_of_line else []
                j = i + 1
                
                while j < len(lines):
                    next_line = lines[j].strip()
                    # Stop if we hit another date
                    if re.match(r'^\d{2}\s+[A-Za-z]{3},\s+\d{4}', next_line):
                        break
                    # Stop if empty line or page footer
                    if not next_line or 'Page' in next_line or 'Kotak' in next_line:
                        j += 1
                        continue
                    transaction_lines.append(next_line)
                    j += 1
                
                # Join all transaction lines
                full_text = ' '.join(transaction_lines)
                
                # Extract amounts - look for patterns like "+54,533.00" or "-300.00" or "1,035.93"
                # Kotak uses +/- prefix for credit/debit
                amounts = re.findall(r'([+-]?[\d,]+\.\d{2})', full_text)
                
                transaction_details = ''
                reference = ''
                debit = ''
                credit = ''
                balance = ''
                
                # Logic: Last number is always balance
                # Numbers with - prefix are debits
                # Numbers with + prefix are credits
                if len(amounts) >= 1:
                    balance = amounts[-1].replace('+', '').replace('-', '')
                    
                    # Remove balance from text
                    full_text = full_text.replace(amounts[-1], '', 1).strip()
                    
                    # Process remaining amounts
                    for amt in amounts[:-1]:
                        if amt.startswith('-'):
                            debit = amt.replace('-', '')
                            full_text = full_text.replace(amt, '', 1).strip()
                        elif amt.startswith('+'):
                            credit = amt.replace('+', '')
                            full_text = full_text.replace(amt, '', 1).strip()
                
                # Extract reference number (UPI-XXXXXXXXXX or similar patterns)
                ref_match = re.search(r'(UPI-\d+|FCM-[\w-]+|[A-Z]+-\d+)', full_text)
                if ref_match:
                    reference = ref_match.group(1)
                    full_text = full_text.replace(reference, '', 1).strip()
                
                # Remaining text is transaction details
                transaction_details = re.sub(r'\s+', ' ', full_text).strip()
                
                transactions.append([date, transaction_details, reference, debit, credit, balance])
                i = j
            else:
                i += 1
        
        logger.info(f"Extracted {len(transactions)} transactions from page {page_num}")
        return transactions

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
            
            is_continuation = (date_val == "" or date_val.upper() == "OPENING BALANCE")
            
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
                if date_val and date_val.upper() != "OPENING BALANCE":
                    last_valid_idx = i
                elif last_valid_idx == -1:
                    rows_to_drop.append(i)

        if rows_to_drop:
            logger.info(f"Merging {len(rows_to_drop)} continuation rows.")
            df = df.drop(rows_to_drop).reset_index(drop=True)

        # Fix Dates - Kotak uses format like "01 Mar, 2023"
        for col in ['DATE']:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str).replace('nan', '').replace('<NA>', '')
            
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # Parse date format "01 Mar, 2023"
            parsed_dates = pd.to_datetime(df[col], format='%d %b, %Y', errors='coerce')
            
            df[col] = parsed_dates.dt.date

        # Convert Numbers
        num_cols = ['DEBIT', 'CREDIT', 'BALANCE']
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with 0
        for col in ['DEBIT', 'CREDIT']:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Rename columns to standard format
        df = df.rename(columns={
            'DATE': 'Date',
            'TRANSACTION DETAILS': 'Particulars',
            'CHEQUE/REFERENCE#': 'Reference',
            'CREDIT': 'Credit',
            'DEBIT': 'Debit',
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
    pdf_file = r'.\Statements\KOTAK.pdf'
    output_excel = r'.\Statements\KOTAK.xlsx'
    
    password = input("Enter PDF password (press Enter if not password-protected): ").strip()
    
    try:
        extractor = KotakStatementExtractor(pdf_file, password=password)
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
