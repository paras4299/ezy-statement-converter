# Ezy Statement Converter

A beautiful, professional web application to convert bank statement PDFs to Excel format. Supports HDFC, SBI, Axis, ICICI, Kotak, PNB, and Union Bank.

## Features

- 🚀 **Fast Conversion** - Convert statements in seconds
- 🔒 **Secure** - Files are processed securely and deleted immediately
- 🎨 **Beautiful UI** - Modern, responsive design
- 📱 **Mobile Friendly** - Works on all devices
- 🏦 **Multiple Banks** - Support for HDFC, SBI, Axis, ICICI, Kotak, PNB, and Union Bank
- 📊 **Accurate** - Advanced parsing algorithms for precise data extraction

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Create necessary folders:
```bash
mkdir uploads outputs
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Select your bank from the dropdown
2. Upload your PDF bank statement
3. Click "Convert to Excel"
4. Download your Excel file

## Supported Banks

- HDFC Bank
- State Bank of India (SBI)
- Axis Bank
- ICICI Bank
- Kotak Mahindra Bank
- Punjab National Bank (PNB)
- Union Bank of India

All banks support password-protected PDFs.

## Project Structure

```
.
├── app.py                 # Flask application
├── hdfc.py               # HDFC Bank extractor
├── sbi.py                # SBI Bank extractor
├── axis.py               # Axis Bank extractor
├── icici.py              # ICICI Bank extractor
├── kotak.py              # Kotak Bank extractor
├── pnb.py                # PNB Bank extractor
├── union.py              # Union Bank extractor
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── about.html
│   ├── contact.html
│   ├── privacy.html
│   └── terms.html
├── uploads/              # Temporary upload folder
├── outputs/              # Converted files folder
└── requirements.txt      # Python dependencies
```

## Security

- All uploaded files are processed securely
- Files are automatically deleted after conversion
- No data is stored on the server
- Encrypted transmission

## Technologies Used

- **Backend**: Flask (Python)
- **PDF Processing**: pdfplumber, camelot-py
- **Data Processing**: pandas
- **Frontend**: HTML5, CSS3, JavaScript
- **Icons**: Font Awesome
- **Fonts**: Google Fonts (Inter)

## License

MIT License

## Support

For support, email support@ezystatementconverter.com or visit our contact page.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
