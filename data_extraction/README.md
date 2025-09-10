# PDF Text Extraction Pipeline with PostgreSQL

A simple, efficient PDF processing system that converts PDFs to images, extracts text using OpenRouter's GPT-4o-mini vision model, and stores the results in a PostgreSQL database.

## 🚀 Features

- **PDF to Image Conversion**: High-quality PNG image conversion using PyMuPDF
- **Advanced OCR**: OpenRouter GPT-4o-mini vision model for accurate text extraction
- **PostgreSQL Integration**: Structured storage with document and page content tables
- **Batch Processing**: Process single files or entire directories
- **Real-time Progress**: Detailed logging and progress tracking
- **No Web Interface**: Pure command-line tool for simplicity

## 🏗️ Architecture

### Pipeline Flow
```
PDF Upload → Image Conversion → OCR Processing → Database Storage → File Output
```

### Database Schema
- **`documents`**: Document metadata (id, name, filename, total_pages, timestamps)
- **`pages_content`**: Page-level content (document_id, page_number, page_content)

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL 12+
- OpenRouter API key

## 🔧 Installation

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd Insurance-Agent-RAG-Chatbot-main
```

### 2. Install Dependencies
```bash
pip install -r simple_requirements.txt
```

### 3. Database Setup
```bash
# Create PostgreSQL database
createdb pdf_processor

# Setup database tables
python setup_database.py
```

### 4. Configure Environment
```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-v1-your-api-key-here"

# Set database URL (optional)
export DATABASE_URL="postgresql://username:password@localhost:5432/pdf_processor"
```

## 🚀 Usage

### Quick Start
```bash
# Process a single PDF
python example_usage.py

# Process multiple PDFs
python batch_processor.py

# View database contents
python view_database.py
```

### Command Line Options
```bash
# Process single PDF
python simple_pdf_processor.py "document.pdf" --api-key "your-key"

# Process directory of PDFs
python simple_pdf_processor.py "./pdfs/" --output "./results" --api-key "your-key"

# With verbose logging
python simple_pdf_processor.py "document.pdf" --verbose
```

## 📁 Project Structure

```
Insurance-Agent-RAG-Chatbot-main/
├── simple_pdf_processor.py    # Main processor class
├── example_usage.py           # Single PDF processing example
├── batch_processor.py         # Multiple PDF processing
├── setup_database.py          # Database initialization
├── view_database.py           # Database content viewer
├── simple_requirements.txt    # Python dependencies
├── README.md                  # This file
└── pdf_output/               # Output directory
    ├── images/               # Converted page images
    └── texts/                # Extracted text files
```

## 🗄️ Database Schema

### Documents Table
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path TEXT,
    total_pages INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### Pages Content Table
```sql
CREATE TABLE pages_content (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    page_content TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## 🔧 Configuration

### Environment Variables
```bash
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
DATABASE_URL=postgresql://username:password@localhost:5432/pdf_processor
```

### API Configuration
- **Model**: `openai/gpt-4o-mini`
- **Endpoint**: `https://openrouter.ai/api/v1/chat/completions`
- **Image Format**: Base64 encoded PNG
- **Max Tokens**: 4000
- **Temperature**: 0.1

## 📊 Output Structure

### File Output
```
pdf_output/
├── images/
│   └── DocumentName/
│       ├── page_001.png
│       ├── page_002.png
│       └── ...
└── texts/
    ├── DocumentName_page_001.txt
    ├── DocumentName_page_002.txt
    ├── DocumentName_combined.txt
    └── DocumentName_metadata.json
```

### Database Output
- **Documents**: Metadata stored in `documents` table
- **Page Content**: Text content stored in `pages_content` table
- **Relationships**: Foreign key linking pages to documents

## 🛠️ API Endpoints

### OpenRouter Integration
- **Vision Model**: GPT-4o-mini for OCR processing
- **Rate Limiting**: Built-in request management
- **Error Handling**: Graceful failure recovery
- **Cost Optimization**: Efficient token usage

## 📈 Performance

### Processing Speed
- **Image Conversion**: ~1-2 seconds per page
- **OCR Processing**: ~3-10 seconds per page (depending on content)
- **Database Storage**: ~0.1 seconds per page
- **Total**: ~5-15 seconds per page

### Scalability
- **Batch Processing**: Process multiple PDFs simultaneously
- **Database Indexing**: Optimized queries for large datasets
- **Memory Efficient**: Stream processing for large files

## 🔍 Troubleshooting

### Common Issues

**1. Database Connection Error**
```bash
# Check PostgreSQL is running
pg_ctl status

# Verify database exists
psql -l | grep pdf_processor
```

**2. OpenRouter API Errors**
```bash
# Check API key
echo $OPENROUTER_API_KEY

# Test API connection
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models
```

**3. PDF Processing Issues**
```bash
# Check file permissions
ls -la document.pdf

# Verify PDF is not corrupted
file document.pdf
```

### Debug Mode
```bash
# Enable verbose logging
python simple_pdf_processor.py "document.pdf" --verbose
```

## 📝 Examples

### Process Single PDF
```python
from simple_pdf_processor import SimplePDFProcessor
import asyncio

async def main():
    processor = SimplePDFProcessor(
        openrouter_api_key="your-key",
        database_url="postgresql://user:pass@localhost:5432/db"
    )
    
    success = await processor.process_pdf("document.pdf")
    print(f"Processing {'successful' if success else 'failed'}")

asyncio.run(main())
```

### Process Multiple PDFs
```python
pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = {}

for pdf_file in pdf_files:
    success = await processor.process_pdf(pdf_file)
    results[pdf_file] = success

print(f"Processed {sum(results.values())}/{len(pdf_files)} files")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenRouter** for providing access to GPT-4o-mini
- **PyMuPDF** for excellent PDF processing capabilities
- **PostgreSQL** for robust database storage
- **AsyncPG** for efficient async database operations

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation

---

**Note**: Make sure to replace placeholder values in configuration files with your actual credentials and settings.
