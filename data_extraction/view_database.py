#!/usr/bin/env python3
"""
View database contents
"""

import asyncio
import asyncpg
import os

async def view_database():
    """View database contents"""
    
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/pdf_processor')
    
    try:
        conn = await asyncpg.connect(database_url)
        
        # Get all documents
        documents = await conn.fetch('SELECT * FROM documents ORDER BY created_at DESC')
        
        print("📄 DOCUMENTS:")
        print("=" * 80)
        for doc in documents:
            print(f"ID: {doc['id']}")
            print(f"Name: {doc['name']}")
            print(f"Filename: {doc['filename']}")
            print(f"Total Pages: {doc['total_pages']}")
            print(f"Created: {doc['created_at']}")
            print("-" * 40)
        
        # Get page counts for each document
        print("\n📊 PAGE COUNTS:")
        print("=" * 80)
        page_counts = await conn.fetch('''
            SELECT d.name, d.filename, COUNT(pc.id) as page_count
            FROM documents d
            LEFT JOIN pages_content pc ON d.id = pc.document_id
            GROUP BY d.id, d.name, d.filename
            ORDER BY d.created_at DESC
        ''')
        
        for row in page_counts:
            print(f"{row['name']} ({row['filename']}): {row['page_count']} pages")
        
        # Get sample page content
        if documents:
            doc_id = documents[0]['id']
            sample_pages = await conn.fetch('''
                SELECT page_number, LEFT(page_content, 100) as content_preview
                FROM pages_content 
                WHERE document_id = $1 
                ORDER BY page_number 
                LIMIT 3
            ''', doc_id)
            
            print(f"\n📝 SAMPLE CONTENT (Document ID: {doc_id}):")
            print("=" * 80)
            for page in sample_pages:
                print(f"Page {page['page_number']}: {page['content_preview']}...")
                print("-" * 40)
        
        await conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing database: {e}")

if __name__ == "__main__":
    asyncio.run(view_database())
