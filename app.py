from flask import Flask, render_template, request, send_file, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
import traceback

# Import the extractors
from hdfc import CamelotTableExtractor
from sbi import SBIStatementExtractor
from axis import AxisStatementExtractor
from pnb import PNBStatementExtractor
from union import UnionStatementExtractor
from icici import ICICIStatementExtractor
from kotak import KotakStatementExtractor

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}
SUPPORTED_BANKS = {
    'hdfc': 'HDFC Bank',
    'sbi': 'State Bank of India',
    'axis': 'Axis Bank',
    'icici': 'ICICI Bank',
    'kotak': 'Kotak Mahindra Bank',
    'pnb': 'Punjab National Bank',
    'union': 'Union Bank of India'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', banks=SUPPORTED_BANKS)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/convert', methods=['POST'])
def convert():
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        bank = request.form.get('bank')
        password = request.form.get('password', '')  # Get password if provided
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not bank or bank not in SUPPORTED_BANKS:
            return jsonify({'error': 'Invalid bank selected'}), 400
        
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"{timestamp}_{filename}"
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
            
            # Save uploaded file
            file.save(pdf_path)
            logger.info(f"File saved: {pdf_path}")
            
            # Generate output filename
            excel_filename = f"{timestamp}_{bank.upper()}_statement.xlsx"
            excel_path = os.path.join(app.config['OUTPUT_FOLDER'], excel_filename)
            
            # Process based on bank
            try:
                if bank == 'hdfc':
                    extractor = CamelotTableExtractor(pdf_path, password=password)
                    df, summary_df = extractor.extract_tables()
                    if not df.empty:
                        df, summary_df = extractor.clean_dataframe(df, summary_df)
                        extractor.save_to_excel(df, summary_df, excel_path)
                    else:
                        raise Exception("No data extracted from PDF")
                
                elif bank == 'sbi':
                    extractor = SBIStatementExtractor(pdf_path, password=password)
                    df = extractor.extract_tables()
                    if not df.empty:
                        df = extractor.clean_dataframe(df)
                        extractor.save_to_excel(df, excel_path)
                    else:
                        raise Exception("No data extracted from PDF")
                
                elif bank == 'axis':
                    extractor = AxisStatementExtractor(pdf_path, password=password)
                    df = extractor.extract_tables()
                    if not df.empty:
                        df = extractor.clean_dataframe(df)
                        extractor.save_to_excel(df, excel_path)
                    else:
                        raise Exception("No data extracted from PDF")
                
                elif bank == 'icici':
                    extractor = ICICIStatementExtractor(pdf_path, password=password)
                    df = extractor.extract_tables()
                    if not df.empty:
                        df = extractor.clean_dataframe(df)
                        extractor.save_to_excel(df, excel_path)
                    else:
                        raise Exception("No data extracted from PDF")
                
                elif bank == 'kotak':
                    extractor = KotakStatementExtractor(pdf_path, password=password)
                    df = extractor.extract_tables()
                    if not df.empty:
                        df = extractor.clean_dataframe(df)
                        extractor.save_to_excel(df, excel_path)
                    else:
                        raise Exception("No data extracted from PDF")
                
                elif bank == 'pnb':
                    extractor = PNBStatementExtractor(pdf_path, password=password)
                    df = extractor.extract_tables()
                    if not df.empty:
                        df = extractor.clean_dataframe(df)
                        extractor.save_to_excel(df, excel_path)
                    else:
                        raise Exception("No data extracted from PDF")
                
                elif bank == 'union':
                    extractor = UnionStatementExtractor(pdf_path, password=password)
                    df = extractor.extract_tables()
                    if not df.empty:
                        df = extractor.clean_dataframe(df)
                        extractor.save_to_excel(df, excel_path)
                    else:
                        raise Exception("No data extracted from PDF")
                
                # Clean up uploaded file
                os.remove(pdf_path)
                
                logger.info(f"Conversion successful: {excel_path}")
                return jsonify({
                    'success': True,
                    'message': 'Conversion successful!',
                    'download_url': f'/download/{excel_filename}'
                })
            
            except Exception as e:
                logger.error(f"Conversion error: {str(e)}")
                logger.error(traceback.format_exc())
                # Clean up files
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                return jsonify({'error': f'Conversion failed: {str(e)}'}), 500
        
        return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': 'Download failed'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
