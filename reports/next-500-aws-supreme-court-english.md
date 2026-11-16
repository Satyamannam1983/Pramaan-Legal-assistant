# Next 500-PDF batch pre-ingestion report

Prepared: 2026-11-15T18:11:39.799923+00:00

## Manifest

- CSV: `reports/next-500-aws-supreme-court-english.csv`
- Rows: 500; unique document IDs: 500; unique SHA-256 hashes: 500
- Manifest SHA-256: `694d4d5632cb8be53afcb2695bc0e8572e6cc8ab62c980cc04f6a96fda2660dc`
- Scope: already-extracted AWS Supreme Court English PDFs only; no High Court data was downloaded or selected.
- Selection follows the existing pipeline's sorted year/path order and first-seen document-ID rule; the selected range is `2010_11_578_583_EN.pdf` through `2010_4_721_732_EN.pdf`.
- Selected by year: 2010: 500
- Existing MongoDB document IDs excluded: 110; source files mapped and hashed: 110.
- Existing-content SHA-256 exclusion: 110 indexed-source hashes. Duplicate-content candidates skipped while selecting: 0; invalid/empty candidates skipped: 0.
- Selected source PDFs occupy 347,104,484 bytes (331.02 MiB) already on disk; the batch does not copy or modify them.

## Read-only MongoDB baseline

- URI/database: `mongodb://127.0.0.1:27017/pramaan` (ping succeeded)
- Documents: 110; chunks: 4012
- Duplicate document IDs: 0; duplicate chunk IDs: 0; orphan chunks: 0
- Unique document and chunk ID indexes are present. No collections or records were changed.

## Disk estimate

Estimate uses the measured prior 100-PDF batch growth: 55,401,253 logical data bytes, 42,209,280 allocated storage bytes, and 225,280 index bytes. Linear estimate for 500 PDFs is 277,006,265 logical bytes (264.17 MiB), 211,046,400 allocated bytes (201.27 MiB), and 1,126,400 index bytes (1.07 MiB). With a 50% reserve across storage and indexes, budget approximately 318,259,200 bytes (303.51 MiB).

- Available on repository drive: 39,731,728,384 bytes (37.00 GiB).
- Available on MongoDB storage volume per `dbStats`: 39,731,728,384 bytes (37.00 GiB).
- The 347,104,484 source bytes are already extracted locally; they are not an incremental ingestion-space requirement. Actual MongoDB growth can differ from this linear estimate.

## Safety and checkpoint

- This is a manifest only; ingestion was not started.
- Existing `ingestion_status.json` SHA-256 before preparation: `d99d1a30fec01a47897fdbfb996dd375deb490f8f356a3d608e2b002295bc02a`.
- The existing checkpoint and completed-batch manifests were read only and left unchanged.
- Do not use the unrestricted ingestion CLI for this manifest. Await explicit authorization before indexing these exact 500 entries.
