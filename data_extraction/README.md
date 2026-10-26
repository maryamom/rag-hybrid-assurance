# OCR of the scanned PDFs

The hackathon documents are scanned PDFs in French and Arabic, often of poor quality, where classic OCR gives
unusable text. Each page is therefore rendered as an image and transcribed by a vision model through OpenRouter.

```bash
python data_extraction/simple_pdf_processor.py data/raw/pdfs        # a folder (sub-folders included)
python data_extraction/simple_pdf_processor.py data/raw/pdfs/auto/cg.pdf   # one file
```

| Step | Detail |
| --- | --- |
| Render | PyMuPDF renders each page as a PNG at 2× zoom |
| Transcribe | the image is sent to `OCR_MODEL` (default `openai/gpt-4o-mini`), asked for a faithful transcription in the original language |
| Store | `documents` (one row per PDF) and `pages_content` (one row per page) in PostgreSQL |

Settings come from `.env`: `OPENROUTER_API_KEY` (required here), `OCR_MODEL`, and the `POSTGRES_*` values
(or `DATABASE_URL`). Next step: `python -m ingestion.clean_documents`.
