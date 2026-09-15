Start-Process powershell -ArgumentList "-NoExit", "-Command", "uvicorn app.main:app --reload --app-dir backend"
Set-Location frontend
npm run dev
