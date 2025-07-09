import fitz  # PyMuPDF
 
def extract_two_column_text(pdf_path, column_split=None):
    sections = []
 
    try:
        pdf_document = fitz.open(pdf_path)
 
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            blocks = page.get_text("blocks")
 
            # Determine page width and split point if not provided
            page_width = page.rect.width
            split_x = column_split if column_split else page_width / 2
 
            # Separate blocks into left and right columns
            left_column = []
            right_column = []
 
            for block in blocks:
                x0, y0, x1, y1, text, *_ = block
                if text.strip():  # skip empty
                    if x1 <= split_x:
                        left_column.append(block)
                    elif x0 >= split_x:
                        right_column.append(block)
                    else:
                        # If it overlaps the split, assign by center point
                        center_x = (x0 + x1) / 2
                        (left_column if center_x < split_x else right_column).append(block)
 
            # Sort each column by vertical position (y0)
            left_column.sort(key=lambda b: b[1])
            right_column.sort(key=lambda b: b[1])
 
            page_sections = [f"\n--- Page {page_num + 1} ---"]
 
            for block in left_column + right_column:
                text = block[4].strip()
                if text:
                    page_sections.append(text)
 
            sections.extend(page_sections)
 
        pdf_document.close()
        return "\n\n".join(sections)
 
    except Exception as e:
        return f"Error reading PDF: {e}"
 
# Example usage
pdf_file_path = "mir.pdf"  # replace with your PDF file
extracted_sections = extract_two_column_text(pdf_file_path)
 
# Save to a text file
with open("two_column_text.txt", "w", encoding="utf-8") as text_file:
    text_file.write(extracted_sections)
 
print("Two-column text extraction complete. See 'two_column_text.txt'.")