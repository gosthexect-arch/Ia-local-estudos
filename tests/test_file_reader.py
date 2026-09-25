import zipfile

import pytest

from core.file_reader import UnreadableFile, extract_text, image_to_data_uri, sniff_type


def test_text_encodings(tmp_path):
    (tmp_path / "utf8.txt").write_text("ação ñ", encoding="utf-8")
    (tmp_path / "latin.txt").write_bytes("ação e coração".encode("cp1252"))
    (tmp_path / "semext").write_text("sem extensão, mas texto", encoding="utf-8")
    assert extract_text(tmp_path / "utf8.txt") == "ação ñ"
    assert "coração" in extract_text(tmp_path / "latin.txt")
    assert "sem extensão" in extract_text(tmp_path / "semext")


def test_html_and_csv(tmp_path):
    (tmp_path / "p.html").write_text("<html><head><style>x{}</style></head><body><h1>Título</h1><p>Olá</p>"
                                     "<script>bad()</script></body></html>", encoding="utf-8")
    txt = extract_text(tmp_path / "p.html")
    assert "Título" in txt and "Olá" in txt and "bad()" not in txt
    (tmp_path / "d.csv").write_text("a;b\n1;2\n", encoding="utf-8")
    assert extract_text(tmp_path / "d.csv").splitlines()[1] == "1\t2"


def test_office_formats(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Parágrafo do Word")
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "c1", "c2"
    d.save(tmp_path / "a.docx")
    assert "Parágrafo do Word" in extract_text(tmp_path / "a.docx")
    assert "c1 | c2" in extract_text(tmp_path / "a.docx")

    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.append(["nome", "valor"])
    wb.active.append(["x", 42])
    wb.save(tmp_path / "a.xlsx")
    assert "x\t42" in extract_text(tmp_path / "a.xlsx")

    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Slide título"
    prs.save(tmp_path / "a.pptx")
    assert "Slide título" in extract_text(tmp_path / "a.pptx")


def test_pdf_with_text_and_without(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(tmp_path / "vazio.pdf", "wb") as f:
        writer.write(f)
    with pytest.raises(UnreadableFile) as e:
        extract_text(tmp_path / "vazio.pdf")
    assert "OCR" in e.value.reason or "escaneado" in e.value.reason
    assert "pdftotext" in e.value.hint


def test_zip_listing_and_odt(tmp_path):
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as z:
        z.writestr("pasta/nota.txt", "oi")
    assert "pasta/nota.txt" in extract_text(tmp_path / "a.zip")
    with zipfile.ZipFile(tmp_path / "a.odt", "w") as z:
        z.writestr("content.xml", "<office:text><text:p>Texto ODT</text:p></office:text>")
    assert "Texto ODT" in extract_text(tmp_path / "a.odt")


def test_binary_is_unreadable_with_hint(tmp_path):
    (tmp_path / "x.doc").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(range(256)) * 20)
    with pytest.raises(UnreadableFile) as e:
        extract_text(tmp_path / "x.doc")
    assert "soffice" in e.value.hint
    assert "OLE2" in sniff_type(tmp_path / "x.doc")


def test_image_to_data_uri_resizes(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGBA", (3000, 1500), (255, 0, 0, 128)).save(tmp_path / "big.png")
    uri = image_to_data_uri(tmp_path / "big.png", max_side=1000)
    assert uri.startswith("data:image/jpeg;base64,")
    import base64
    import io
    im = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert max(im.size) == 1000 and im.mode == "RGB"
