#!/usr/bin/env python3
"""
Batch PDF Processor - Process multiple PDFs automatically
"""

import asyncio
import os
from pathlib import Path
from simple_pdf_processor import SimplePDFProcessor

async def main():
    # Initialize processor with your OpenRouter API key
    processor = SimplePDFProcessor(
        openrouter_api_key="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",
        output_dir="./pdf_output"
    )
    
    # Option 1: Process specific PDF files
    pdf_files = [
        "CG HANA.pdf",
        # Add more PDF files here
    ]
    
    # Option 2: Automatically find all PDF files in current directory
    # Uncomment the lines below to process all PDFs in the current directory
    # pdf_files = list(Path(".").glob("*.pdf"))
    # pdf_files = [str(f) for f in pdf_files]
    
    # Option 3: Process PDFs from a specific directory
    # pdf_directory = "./my_pdfs/"
    # pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    # pdf_files = [str(f) for f in pdf_files]
    
    if not pdf_files:
        print("❌ No PDF files found to process!")
        return
    
    print(f"🚀 Found {len(pdf_files)} PDF files to process:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf_file}")
    
    print("\n" + "="*50)
    
    results = {}
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 Processing {i}/{len(pdf_files)}: {pdf_file}")
        
        # Check if file exists
        if not os.path.exists(pdf_file):
            print(f"❌ File not found: {pdf_file}")
            results[pdf_file] = False
            continue
        
        try:
            success = await processor.process_pdf(pdf_file)
            results[pdf_file] = success
            
            if success:
                print(f"✅ {pdf_file} processed successfully!")
            else:
                print(f"❌ {pdf_file} processing failed!")
                
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
            results[pdf_file] = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 PROCESSING SUMMARY")
    print("="*50)
    
    successful = sum(1 for success in results.values() if success)
    failed = len(pdf_files) - successful
    
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(successful/len(pdf_files)*100):.1f}%")
    
    print("\nDetailed Results:")
    for pdf_file, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {pdf_file}: {status}")
    
    if successful > 0:
        print(f"\n📁 Check the './pdf_output' folder for results:")
        print("   - images/ : Converted page images")
        print("   - texts/  : Extracted text files")

if __name__ == "__main__":
    asyncio.run(main())
