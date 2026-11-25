from src.data.convert import voc_to_yolo

XML = """<annotation><size><width>720</width><height>720</height></size>
<object><name>D00</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>360</xmax><ymax>72</ymax></bndbox></object>
<object><name>D40</name><bndbox><xmin>700.5</xmin><ymin>600</ymin><xmax>800</xmax><ymax>800</ymax></bndbox></object>
<object><name>D44</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
<object><name>D10</name><bndbox><xmin>900</xmin><ymin>10</ymin><xmax>950</xmax><ymax>20</ymax></bndbox></object>
</annotation>"""


def test_voc_to_yolo() -> None:
    w, h, lines, dropped = voc_to_yolo(XML)
    assert (w, h) == (720, 720)
    assert lines[0] == "0 0.250000 0.050000 0.500000 0.100000"
    cls, cx, cy, bw, bh = lines[1].split()  # pothole clipped to image edge
    assert cls == "3" and float(cx) + float(bw) / 2 <= 1 + 1e-6 and float(cy) + float(bh) / 2 <= 1 + 1e-6
    assert len(lines) == 2
    assert dropped == {"D44": 1, "<degenerate>": 1}  # D10 box fully outside image
