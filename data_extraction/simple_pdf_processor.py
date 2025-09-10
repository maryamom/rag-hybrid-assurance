#!/usr/bin/env python3
"""
Simple PDF to Text Processor
Converts PDFs to images, extracts text using OCR, and saves to files.
No web interface - just a simple command-line tool.
"""

import os
import sys
import json
import base64
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
import io
import httpx
import asyncpg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePDFProcessor:
    def __init__(self, openrouter_api_key: str = None, output_dir: str = "./output", 
                 database_url: str = None):
        """
        Initialize the PDF processor.
        
        Args:
            openrouter_api_key: OpenRouter API key
            output_dir: Directory to save outputs
            database_url: PostgreSQL database URL
        """
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        self.database_url = database_url or os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/pdf_processor')
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.texts_dir = self.output_dir / "texts"
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.texts_dir.mkdir(exist_ok=True)
        
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"📁 Images directory: {self.images_dir}")
        logger.info(f"📁 Texts directory: {self.texts_dir}")
        logger.info(f"🗄️ Database URL: {self.database_url}")

    async def get_db_connection(self):
        """Get database connection"""
        try:
            conn = await asyncpg.connect(self.database_url)
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    async def init_database(self):
        """Initialize database tables"""
        try:
            conn = await self.get_db_connection()
            
            # Documents table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    file_path TEXT,
                    total_pages INTEGER DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Pages content table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS pages_content (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL,
                    page_content TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_pages_content_document_id 
                ON pages_content(document_id)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_pages_content_page_number 
                ON pages_content(page_number)
            ''')
            
            await conn.close()
            logger.info("✅ Database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise

    async def save_document_to_db(self, pdf_name: str, filename: str, file_path: str, total_pages: int) -> int:
        """Save document to database and return document ID"""
        try:
            conn = await self.get_db_connection()
            
            # Check if document already exists
            existing = await conn.fetchrow('SELECT id FROM documents WHERE filename = $1', filename)
            if existing:
                # Update existing document
                await conn.execute('''
                    UPDATE documents 
                    SET name = $1, file_path = $2, total_pages = $3, updated_at = CURRENT_TIMESTAMP
                    WHERE filename = $4
                ''', pdf_name, file_path, total_pages, filename)
                document_id = existing['id']
                logger.info(f"✅ Updated existing document: {filename} (ID: {document_id})")
            else:
                # Insert new document
                document_id = await conn.fetchval('''
                    INSERT INTO documents (name, filename, file_path, total_pages)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                ''', pdf_name, filename, file_path, total_pages)
                logger.info(f"✅ Created new document: {filename} (ID: {document_id})")
            
            await conn.close()
            return document_id
            
        except Exception as e:
            logger.error(f"❌ Error saving document to database: {e}")
            raise

    async def save_page_content_to_db(self, document_id: int, page_number: int, page_content: str):
        """Save page content to database"""
        try:
            conn = await self.get_db_connection()
            
            # Check if page already exists
            existing = await conn.fetchrow('''
                SELECT id FROM pages_content 
                WHERE document_id = $1 AND page_number = $2
            ''', document_id, page_number)
            
            if existing:
                # Update existing page
                await conn.execute('''
                    UPDATE pages_content 
                    SET page_content = $1, created_at = CURRENT_TIMESTAMP
                    WHERE document_id = $2 AND page_number = $3
                ''', page_content, document_id, page_number)
                logger.info(f"✅ Updated page {page_number} for document {document_id}")
            else:
                # Insert new page
                await conn.execute('''
                    INSERT INTO pages_content (document_id, page_number, page_content)
                    VALUES ($1, $2, $3)
                ''', document_id, page_number, page_content)
                logger.info(f"✅ Saved page {page_number} for document {document_id}")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error saving page content to database: {e}")
            raise

    async def call_openrouter_ocr(self, image_base64: str, page_num: int) -> str:
        """Call OpenRouter API for text extraction using vision model."""
        try:
            if not self.openrouter_api_key:
                logger.warning("⚠️ No OpenRouter API key provided. Using fallback text extraction.")
                return ""
            
            logger.info(f"🔍 Calling OpenRouter OCR for page {page_num + 1}...")
            
            payload = {
                "model": "openai/gpt-4o-mini",  # Using GPT-4o-mini for vision
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract all text from this image. Return only the text content without any additional descriptions or formatting."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]}
                ],
                "max_tokens": 4000,
                "temperature": 0.1
            }
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/your-repo",  # Optional: replace with your app
                        "X-Title": "PDF Text Extractor"  # Optional: replace with your app name
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"❌ OpenRouter API error: {response.status_code} - {response.text}")
                    return ""
                
                result = response.json()
                
                # Extract text from response
                extracted_text = ""
                if 'choices' in result and len(result['choices']) > 0:
                    choice = result['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        extracted_text = choice['message']['content'].strip()
                
                if extracted_text:
                    logger.info(f"✅ Extracted text from page {page_num + 1}: {len(extracted_text)} characters")
                    return extracted_text
                else:
                    logger.warning(f"⚠️ No text extracted from page {page_num + 1}")
                    return ""
                    
        except Exception as e:
            logger.error(f"❌ OpenRouter OCR error for page {page_num + 1}: {e}")
            return ""

    def convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF to images and save them."""
        try:
            logger.info(f"🔄 Converting PDF to images: {pdf_path}")
            
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            image_paths = []
            
            # Create images directory for this PDF
            pdf_name = Path(pdf_path).stem
            pdf_images_dir = self.images_dir / pdf_name
            pdf_images_dir.mkdir(exist_ok=True)
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                
                # Render page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Save image
                image_filename = f"page_{page_num + 1:03d}.png"
                image_path = pdf_images_dir / image_filename
                img.save(image_path, "PNG")
                
                image_paths.append(str(image_path))
                logger.info(f"✅ Page {page_num + 1} saved: {image_path}")
            
            pdf_document.close()
            logger.info(f"✅ PDF converted to {len(image_paths)} images")
            return image_paths
            
        except Exception as e:
            logger.error(f"❌ Error converting PDF to images: {e}")
            return []

    async def extract_text_from_images(self, image_paths: List[str], pdf_name: str) -> List[Dict[str, Any]]:
        """Extract text from images using OCR."""
        try:
            logger.info(f"🔍 Extracting text from {len(image_paths)} images...")
            
            extracted_texts = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    # Read image and convert to base64
                    with open(image_path, "rb") as img_file:
                        image_data = img_file.read()
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    # Call OCR API
                    extracted_text = await self.call_openrouter_ocr(image_base64, i)
                    
                    if extracted_text:
                        extracted_texts.append({
                            "page_number": i + 1,
                            "text_content": extracted_text,
                            "extraction_method": "openrouter_gpt4o_mini",
                            "image_path": image_path
                        })
                        logger.info(f"✅ Page {i + 1} processed: {len(extracted_text)} characters")
                    else:
                        logger.warning(f"⚠️ No text extracted from page {i + 1}")
                        
                except Exception as page_error:
                    logger.error(f"❌ Error processing page {i + 1}: {page_error}")
                    continue
            
            logger.info(f"✅ Text extraction completed: {len(extracted_texts)} pages processed")
            return extracted_texts
            
        except Exception as e:
            logger.error(f"❌ Error in text extraction: {e}")
            return []

    def save_extracted_texts(self, extracted_texts: List[Dict[str, Any]], pdf_name: str):
        """Save extracted texts to files."""
        try:
            # Save individual page texts
            for text_data in extracted_texts:
                page_num = text_data["page_number"]
                text_content = text_data["text_content"]
                
                # Save as individual text file
                text_filename = f"{pdf_name}_page_{page_num:03d}.txt"
                text_path = self.texts_dir / text_filename
                
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                
                logger.info(f"💾 Page {page_num} text saved: {text_path}")
            
            # Save combined text file
            combined_text = "\n\n".join([f"=== PAGE {t['page_number']} ===\n{t['text_content']}" for t in extracted_texts])
            combined_filename = f"{pdf_name}_combined.txt"
            combined_path = self.texts_dir / combined_filename
            
            with open(combined_path, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            
            logger.info(f"💾 Combined text saved: {combined_path}")
            
            # Save metadata JSON
            metadata = {
                "pdf_name": pdf_name,
                "total_pages": len(extracted_texts),
                "extraction_timestamp": datetime.now().isoformat(),
                "extraction_method": "openrouter_gpt4o_mini",
                "pages": extracted_texts
            }
            
            metadata_filename = f"{pdf_name}_metadata.json"
            metadata_path = self.texts_dir / metadata_filename
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Metadata saved: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving extracted texts: {e}")

    async def process_pdf(self, pdf_path: str) -> bool:
        """Process a single PDF file through the complete pipeline."""
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"❌ PDF file not found: {pdf_path}")
                return False
            
            pdf_name = pdf_path.stem
            filename = pdf_path.name
            logger.info(f"🚀 Starting PDF processing: {pdf_name}")
            
            # Initialize database
            await self.init_database()
            
            # Step 1: Convert PDF to images
            image_paths = self.convert_pdf_to_images(str(pdf_path))
            if not image_paths:
                logger.error("❌ Failed to convert PDF to images")
                return False
            
            # Step 2: Save document to database
            document_id = await self.save_document_to_db(pdf_name, filename, str(pdf_path), len(image_paths))
            
            # Step 3: Extract text from images
            extracted_texts = await self.extract_text_from_images(image_paths, pdf_name)
            if not extracted_texts:
                logger.warning("⚠️ No text extracted from any pages")
                return False
            
            # Step 4: Save page contents to database
            for text_data in extracted_texts:
                await self.save_page_content_to_db(
                    document_id, 
                    text_data['page_number'], 
                    text_data['text_content']
                )
            
            # Step 5: Save extracted texts to files (optional)
            self.save_extracted_texts(extracted_texts, pdf_name)
            
            logger.info(f"✅ PDF processing completed successfully: {pdf_name}")
            logger.info(f"📊 Results: {len(extracted_texts)} pages processed, {sum(len(t['text_content']) for t in extracted_texts)} total characters")
            logger.info(f"🗄️ Document saved to database with ID: {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            return False

    async def process_multiple_pdfs(self, pdf_directory: str) -> Dict[str, bool]:
        """Process all PDF files in a directory."""
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            logger.error(f"❌ Directory not found: {pdf_directory}")
            return {}
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"⚠️ No PDF files found in: {pdf_directory}")
            return {}
        
        logger.info(f"📁 Found {len(pdf_files)} PDF files to process")
        
        results = {}
        for pdf_file in pdf_files:
            logger.info(f"🔄 Processing: {pdf_file.name}")
            success = await self.process_pdf(str(pdf_file))
            results[pdf_file.name] = success
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        logger.info(f"📊 Processing complete: {successful}/{len(pdf_files)} PDFs processed successfully")
        
        return results

def main():
    """Main function to run the PDF processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple PDF to Text Processor")
    parser.add_argument("input", help="PDF file or directory containing PDFs")
    parser.add_argument("--output", "-o", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize processor
    processor = SimplePDFProcessor(
        openrouter_api_key=args.api_key,
        output_dir=args.output
    )
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single PDF
        logger.info(f"📄 Processing single PDF: {input_path}")
        success = asyncio.run(processor.process_pdf(str(input_path)))
        sys.exit(0 if success else 1)
    
    elif input_path.is_dir():
        # Process directory of PDFs
        logger.info(f"📁 Processing directory: {input_path}")
        results = asyncio.run(processor.process_multiple_pdfs(str(input_path)))
        sys.exit(0 if all(results.values()) else 1)
    
    else:
        logger.error(f"❌ Input path not found: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
