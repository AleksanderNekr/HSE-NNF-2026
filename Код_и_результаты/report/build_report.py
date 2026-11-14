"""
Сборка итогового отчёта в шаблоне конференции, плюс листинги программного кода
в приложении.

Берётся оригинальный файл информационного письма, из него удаляется всё до
строки «УДК», а каждый абзац отчёта создаётся клонированием соответствующего
абзаца-образца шаблона: свойства абзаца (отступы, выравнивание, интервалы) и
свойства текста (шрифт, кегль) переносятся без изменений, заменяется только
текст. Так формат шаблона сохраняется, как того требует оргкомитет.

Содержание задаётся в report_content.py списком блоков.
"""

from __future__ import annotations

import copy
import re
import shutil
import sys
import zipfile
import os
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_content as C  # noqa: E402

TEMPLATE_NAME = "1_е_Информационное_письмо_ИСКУССТВЕННЫЙ_ИНТЕЛЛЕКТ_В_РЕШЕНИИ_АКТУАЛЬНЫХ.docx"


def _find_template() -> Path:
    """Шаблон конференции: переменная окружения LAB6_TEMPLATE или файл рядом со скриптом."""
    cands = [os.environ.get("LAB6_TEMPLATE", ""), HERE / TEMPLATE_NAME]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(f"Не найден шаблон конференции {TEMPLATE_NAME}: положите его в {HERE} "
                            "или укажите путь в переменной окружения LAB6_TEMPLATE")


TEMPLATE = _find_template()
WORK = HERE / "_build"
OUT = HERE / C.OUTPUT_NAME

EMU_PER_IN = 914400


# --------------------------------------------------------------------------
def load_parts(xml: str):
    head = xml[:xml.index("<w:body>") + len("<w:body>")]
    tail = xml[xml.index("</w:body>"):]
    body = xml[len(head):xml.index("</w:body>")]
    parts = re.findall(r'(<w:p[ >].*?</w:p>|<w:tbl>.*?</w:tbl>|<w:sectPr.*?</w:sectPr>)',
                       body, flags=re.S)
    return head, parts, tail


def ptext(p: str) -> str:
    return "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))


def strip_rsid(x: str) -> str:
    return re.sub(r' w14:(paraId|textId)="[^"]*"| w:rsid\w*="[^"]*"', "", x)


def ppr_of(p: str) -> str:
    m = re.search(r"<w:pPr>.*?</w:pPr>", p, flags=re.S)
    return m.group(0) if m else ""


def base_rpr(p: str) -> str:
    """Свойства текста первого текстового прогона абзаца-образца
    без жирного/курсива — их выставляем сами."""
    m = re.search(r"<w:r>\s*(<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)\s*(?:<w:lastRenderedPageBreak/>)?\s*<w:t",
                  p, flags=re.S)
    rpr = m.group(1) if m else "<w:rPr></w:rPr>"
    rpr = re.sub(r"<w:b/>|<w:bCs/>|<w:i/>|<w:iCs/>|<w:bCs w:val=\"0\"/>|<w:vertAlign[^>]*/>", "", rpr)
    return rpr


def run(text: str, rpr: str, b=False, i=False, sup=False, sub=False) -> str:
    extra = ""
    if b:
        extra += "<w:b/><w:bCs/>"
    if i:
        extra += "<w:i/><w:iCs/>"
    if sup:
        extra += '<w:vertAlign w:val="superscript"/>'
    if sub:
        extra += '<w:vertAlign w:val="subscript"/>'
    r = rpr.replace("<w:rPr>", "<w:rPr>" + extra, 1) if extra else rpr
    return f'<w:r>{r}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


NBSP = "\u00a0"


def typo(s: str) -> str:
    """Типографика: неразрывные пробелы, чтобы формулы и числа с единицами
    не разрывались переносом строки («ξ = 0», «2 %», «№ 4», «С. 48», «6,3 МПа»)."""
    s = re.sub(r" (=|<|>|≤|≥|–|−) (?=\S)", lambda m: NBSP + m.group(1) + " " if m.group(1) == "–" else NBSP + m.group(1) + NBSP, s)
    s = re.sub(r"(?<![\d,])(\d{1,4}) (?=[а-яё])", lambda m: m.group(1) + NBSP, s)   # «15 классических»
    s = re.sub(r"(\d) (%|МПа|с\b|эпох)", lambda m: m.group(1) + NBSP + m.group(2), s)
    s = re.sub(r"(№|С\.|P\.|Т\.|Vol\.) (?=\S)", lambda m: m.group(1) + NBSP, s)
    s = re.sub(r"(т\.) (е\.)", "\\1" + NBSP + "\\2", s)
    return s


def parse_inline(s: str):
    """Мини-разметка: **жирный**, *курсив*, ^{верх}, _{низ}."""
    tokens = re.split(r"(\*\*.+?\*\*|\*[^*]+?\*|\^\{[^}]*\}|_\{[^}]*\})", s)
    out = []
    for t in tokens:
        if not t:
            continue
        if t.startswith("**"):
            out.append((t[2:-2], dict(b=True)))
        elif t.startswith("*"):
            out.append((t[1:-1], dict(i=True)))
        elif t.startswith("^{"):
            out.append((t[2:-1], dict(sup=True)))
        elif t.startswith("_{"):
            out.append((t[2:-1], dict(sub=True)))
        else:
            out.append((t, {}))
    return out


def make_par(proto: str, text: str, all_bold=False, all_italic=False,
             tail_runs: str = "", fix_mark=False) -> str:
    ppr = ppr_of(proto)
    if fix_mark:   # в образце аннотации знак абзаца случайно зачёркнут и подсвечен
        ppr = re.sub(r"<w:strike/>|<w:highlight[^>]*/>|<w:u [^>]*/>", "", ppr)
    rpr = base_rpr(proto)
    text = typo(text)
    runs = "".join(run(t, rpr, b=f.get("b", False) or all_bold,
                       i=f.get("i", False) or all_italic,
                       sup=f.get("sup", False), sub=f.get("sub", False))
                   for t, f in parse_inline(text))
    return f"<w:p>{ppr}{runs}{tail_runs}</w:p>"


def drawing_par(proto: str, rid: str, png: Path, width_in: float, pic_id: int) -> str:
    w, h = Image.open(png).size
    cx = int(width_in * EMU_PER_IN)
    cy = int(cx * h / w)
    p = proto
    p = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>', f'<wp:extent cx="{cx}" cy="{cy}"/>', p)
    p = re.sub(r'<a:ext cx="\d+" cy="\d+"/>', f'<a:ext cx="{cx}" cy="{cy}"/>', p)
    p = re.sub(r'r:embed="[^"]+"', f'r:embed="{rid}"', p)
    p = re.sub(r'<wp:docPr id="\d+" name="[^"]*"/>',
               f'<wp:docPr id="{pic_id}" name="Рисунок {pic_id}"/>', p)
    p = re.sub(r'<wp:effectExtent[^>]*/>', '<wp:effectExtent l="0" t="0" r="0" b="0"/>', p)
    return p


def table_xml(rows: list[list[str]], widths_tw: list[int], font_sz: int = 22) -> str:
    """Простая таблица в стиле текста шаблона (Times New Roman, 12 пт)."""
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths_tw)
    rpr_base = f'<w:rPr><w:sz w:val="{font_sz}"/><w:szCs w:val="{font_sz}"/></w:rPr>'
    trs = []
    for ri, row in enumerate(rows):
        tcs = []
        for ci, cell in enumerate(row):
            jc = "left" if ci == 0 else "center"
            runs = "".join(run(t, rpr_base, b=(ri == 0) or f.get("b", False),
                               i=f.get("i", False), sub=f.get("sub", False),
                               sup=f.get("sup", False))
                           for t, f in parse_inline(typo(cell)))
            tcs.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths_tw[ci]}" w:type="dxa"/>'
                f'<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr><w:jc w:val="{jc}"/><w:rPr><w:sz w:val="{font_sz}"/></w:rPr></w:pPr>{runs}</w:p></w:tc>')
        header = '<w:trPr><w:tblHeader/></w:trPr>' if ri == 0 else ""
        trs.append(f"<w:tr>{header}{''.join(tcs)}</w:tr>")
    b = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    borders = "".join(b.format(x) for x in ["top", "left", "bottom", "right", "insideH", "insideV"])
    return (f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths_tw)}" w:type="dxa"/><w:jc w:val="center"/>'
            f'<w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
            f'<w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
            f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>')


# --------------------------------------------------------------------------
def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    with zipfile.ZipFile(TEMPLATE) as z:
        z.extractall(WORK)
    for p in WORK.rglob("*"):
        if p.is_symlink():
            p.unlink()

    doc_path = WORK / "word" / "document.xml"
    head, parts, tail = load_parts(doc_path.read_text(encoding="utf-8"))
    udk = next(i for i, p in enumerate(parts) if "УДК" in ptext(p))
    P = {k: strip_rsid(parts[udk + off]) for k, off in {
        "udk": 0, "title": 1, "title_last": 2, "author": 3, "affil": 4, "email": 6,
        "abstract": 12, "keywords": 13, "body": 14, "heading": 18, "figure": 24,
        "caption": 29, "empty": 32, "bibhead": 52, "bib": 53,
        "en_title": 66, "en_author": 67, "en_affil": 68, "en_email": 70,
        "en_abs": 75, "en_kw": 76, "end": 77,
    }.items()}
    sect = parts[-1]
    assert "Библиографический список" in ptext(P["bibhead"])
    assert "Рисунок 2" in ptext(P["caption"])

    footref = re.search(r'<w:r><w:rPr><w:rStyle w:val="a6"/>.*?<w:footnoteReference w:id="1"/></w:r>',
                        P["title_last"], flags=re.S).group(0)

    rels_path = WORK / "word" / "_rels" / "document.xml.rels"
    rels = rels_path.read_text(encoding="utf-8")
    media = WORK / "word" / "media"

    out: list[str] = []
    fig_no = 0
    for kind, *args in C.BLOCKS:
        if kind == "udk":
            out.append(make_par(P["udk"], args[0]))
        elif kind == "title":
            lines = args[0]
            for j, line in enumerate(lines):
                last = j == len(lines) - 1
                out.append(make_par(P["title_last"] if last else P["title"], line,
                                    all_bold=True, tail_runs=footref if last else ""))
        elif kind == "author":
            name, affil_lines, email = args
            out.append(make_par(P["author"], name, all_bold=True, all_italic=True))
            for line in affil_lines:
                out.append(make_par(P["affil"], line))
            out.append(make_par(P["email"], email))
        elif kind == "abstract":
            out.append(make_par(P["abstract"], args[0], fix_mark=True))
        elif kind == "keywords":
            out.append(make_par(P["keywords"], "**Ключевые слова**: " + args[0]))
        elif kind == "p":
            out.append(make_par(P["body"], args[0]))
        elif kind == "h":
            out.append(make_par(P["heading"], args[0], all_bold=True))
        elif kind == "fig":
            png, caption, width_in = args
            fig_no += 1
            rid = f"rIdThesisFig{fig_no}"
            name = f"thesis_fig{fig_no}.png"
            shutil.copy(png, media / name)
            rels = rels.replace("</Relationships>",
                                f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                                f'officeDocument/2006/relationships/image" Target="media/{name}"/>'
                                "</Relationships>")
            fp = drawing_par(P["figure"], rid, Path(png), width_in, 900 + fig_no)
            # подпись держится на одной странице с рисунком
            fp = fp.replace("<w:pPr>", "<w:pPr><w:keepNext/>", 1)
            out.append(fp)
            out.append(make_par(P["caption"], f"Рисунок {fig_no}. {caption}", all_bold=True))
        elif kind == "table":
            caption, rows, widths = args
            cap = make_par(P["caption"], caption, all_bold=True)
            cap = cap.replace("<w:pPr>", "<w:pPr><w:keepNext/>", 1)
            cap = re.sub(r'<w:spacing w:after="240"/>', '<w:spacing w:after="80"/>', cap)
            out.append(cap)
            out.append(table_xml(rows, widths))
            out.append(make_par(P["empty"], ""))
        elif kind == "code":
            ppr = ('<w:pPr><w:keepLines/><w:spacing w:before="0" w:after="0" w:line="216" w:lineRule="auto"/>'
                   '<w:ind w:left="0" w:firstLine="0"/><w:jc w:val="left"/></w:pPr>')
            rpr = ('<w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>'
                   '<w:sz w:val="15"/><w:szCs w:val="15"/><w:lang w:val="en-US"/></w:rPr>')
            for line in args[0].strip("\n").split("\n"):
                out.append(f'<w:p>{ppr}<w:r>{rpr}<w:t xml:space="preserve">{escape(line)}</w:t></w:r></w:p>')
            out.append(make_par(P["empty"], ""))
        elif kind == "lcap":
            cp = make_par(P["body"], args[0])
            out.append(cp.replace("<w:pPr>", "<w:pPr><w:keepNext/>", 1))
        elif kind == "pagebreak":
            out.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        elif kind == "empty":
            out.append(make_par(P["empty"], ""))
        elif kind == "bibhead":
            out.append(make_par(P["bibhead"], "Библиографический список", all_bold=True))
        elif kind == "bib":
            out.append(make_par(P["bib"], args[0]))
        elif kind == "en_title":
            out.append(make_par(P["en_title"], args[0], all_bold=True))
        elif kind == "en_author":
            name, affil_lines, email = args
            out.append(make_par(P["en_author"], name, all_bold=True, all_italic=True))
            for line in affil_lines:
                out.append(make_par(P["en_affil"], line))
            out.append(make_par(P["en_email"], email))
        elif kind == "en_abs":
            out.append(make_par(P["en_abs"], args[0]))
        elif kind == "en_kw":
            out.append(make_par(P["en_kw"], "**Keywords**: " + args[0]))
        else:
            raise ValueError(kind)
    out.append(P["end"])
    out.append(sect)

    doc_path.write_text(head + "".join(out) + tail, encoding="utf-8")
    rels_path.write_text(rels, encoding="utf-8")

    # сноска с копирайтом
    fn_path = WORK / "word" / "footnotes.xml"
    fn = fn_path.read_text(encoding="utf-8")
    fn = re.sub(r'(<w:footnote w:id="1">.*?<w:footnoteRef/></w:r>).*?(</w:p></w:footnote>)',
                lambda m: m.group(1) + f'<w:r><w:t xml:space="preserve"> {escape(C.COPYRIGHT)}</w:t></w:r>' + m.group(2),
                fn, flags=re.S)
    fn_path.write_text(fn, encoding="utf-8")

    # удаляем неиспользуемые картинки письма (логотипы) — меньше размер файла
    used = set(re.findall(r'r:embed="([^"]+)"', head + "".join(out)))
    for rid, target in re.findall(r'<Relationship Id="([^"]+)" Type="[^"]+/image" Target="([^"]+)"/>', rels):
        if rid not in used:
            rels = re.sub(rf'<Relationship Id="{rid}" [^>]*/>', "", rels)
            f = WORK / "word" / target
            if f.exists():
                f.unlink()
    rels_path.write_text(rels, encoding="utf-8")

    # метаданные
    core = WORK / "docProps" / "core.xml"
    c = core.read_text(encoding="utf-8")
    c = re.sub(r"<dc:title>.*?</dc:title>|<dc:title/>", f"<dc:title>{escape(C.META_TITLE)}</dc:title>", c)
    c = re.sub(r"<dc:creator>.*?</dc:creator>", f"<dc:creator>{escape(C.META_AUTHOR)}</dc:creator>", c)
    c = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>",
               f"<cp:lastModifiedBy>{escape(C.META_AUTHOR)}</cp:lastModifiedBy>", c)
    core.write_text(c, encoding="utf-8")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(WORK.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(WORK).as_posix())
    print("записан", OUT)


if __name__ == "__main__":
    main()
