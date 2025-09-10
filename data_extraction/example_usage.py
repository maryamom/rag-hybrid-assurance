#!/usr/bin/env python3
"""
Example usage of the Simple PDF Processor
"""

import asyncio
from simple_pdf_processor import SimplePDFProcessor

async def main():
    # Initialize processor with your OpenRouter API key and database
    processor = SimplePDFProcessor(
        openrouter_api_key="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",
        output_dir="./pdf_output",
        database_url="postgresql://postgres:password@localhost:5432/pdf_processor"  # Update with your DB credentials
    )
    
    # List of PDF files to process
    pdf_files = [
        "CG HANA.pdf",
        # Add more PDF files here as needed
        # "document2.pdf",
        # "document3.pdf",
    ]
    
    print(f"🚀 Processing {len(pdf_files)} PDF files...")
    
    results = {}
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 Processing {i}/{len(pdf_files)}: {pdf_file}")
        success = await processor.process_pdf(pdf_file)
        results[pdf_file] = success
        
        if success:
            print(f"✅ {pdf_file} processed successfully!")
        else:
            print(f"❌ {pdf_file} processing failed!")
    
    # Summary
    successful = sum(1 for success in results.values() if success)
    print(f"\n📊 Summary: {successful}/{len(pdf_files)} PDFs processed successfully")
    
    for pdf_file, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {pdf_file}: {status}")

if __name__ == "__main__":
    asyncio.run(main())
