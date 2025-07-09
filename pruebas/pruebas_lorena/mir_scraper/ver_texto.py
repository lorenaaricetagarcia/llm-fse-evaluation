import os
from PyPDF2 import PdfReader

folder = "mir_pdfs"
pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]

for pdf_file in pdf_files[:1]:  # Solo prueba con el primer PDF
    pdf_path = os.path.join(folder, pdf_file)
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    print(f"\n📝 Texto extraído del PDF: {pdf_file}\n")
    print(text[:3000])  # Mostrar solo los primeros 3000 caracteres
