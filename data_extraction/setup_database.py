#!/usr/bin/env python3
"""
Database setup script for PDF Processor
Creates the required PostgreSQL database and tables
"""

import asyncio
import asyncpg
import os

async def setup_database():
    """Setup PostgreSQL database and tables"""
    
    # Database configuration
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/pdf_processor')
    
    print("🗄️ Setting up PostgreSQL database...")
    print(f"Database URL: {database_url}")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(database_url)
        print("✅ Connected to PostgreSQL database")
        
        # Create documents table
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
        print("✅ Created 'documents' table")
        
        # Create pages_content table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS pages_content (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                page_content TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✅ Created 'pages_content' table")
        
        # Create indexes for better performance
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_pages_content_document_id 
            ON pages_content(document_id)
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_pages_content_page_number 
            ON pages_content(page_number)
        ''')
        print("✅ Created database indexes")
        
        # Test the tables
        result = await conn.fetchval('SELECT COUNT(*) FROM documents')
        print(f"✅ Database setup complete! Documents table has {result} records")
        
        await conn.close()
        print("🎉 Database setup successful!")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check your database credentials")
        print("3. Ensure the database 'pdf_processor' exists")
        print("4. Update the DATABASE_URL environment variable if needed")

if __name__ == "__main__":
    asyncio.run(setup_database())
