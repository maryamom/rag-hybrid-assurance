# Data

The BH Assurance hackathon data was shared with every participant. Put it here:

```
data/
  raw/
    tables/           the Excel files, as provided (one table per sheet after loading)
    pdfs/<branche>/   the scanned PDFs, one folder per insurance branch (auto, habitation, ...)
  processed/
    cleaned_docs.jsonl   OCR output after cleaning (written by ingestion.clean_documents --export)
```

The folder name under `pdfs/` becomes the document's branch. PDFs directly in `pdfs/` get the branch `general`.

Commit `processed/cleaned_docs.jsonl`: with it, the indexes can be rebuilt without running the OCR again.
If the PDFs are large (more than about 50 MB in total), store them with Git LFS or link to them instead of committing them.

After loading the tables, run `python -m ingestion.check_data`. It lists, for each piece of information the
pipeline uses, the table and columns it found. If a column has a different name, add it to the candidates
in [`ingestion/schema.py`](../ingestion/schema.py).
