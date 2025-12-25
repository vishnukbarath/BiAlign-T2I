from pathlib import Path
from typing import Iterable, Dict, Any
from PIL import Image
import base64
import io


class HTMLReport:
    """Simple HTML report generator for visualization outputs."""

    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def add_row(self, image: Image.Image, name: str, metadata: Dict[str, Any] = None):
        p = self.out_dir / f"{name}.png"
        image.save(p)
        self.rows.append({"name": name, "image_path": p.name, "metadata": metadata or {}})

    def write(self, filename: str = "report.html"):
        idx = self.out_dir / filename
        with idx.open("w", encoding="utf-8") as f:
            f.write("<html><head><meta charset=\"utf-8\"><title>Evaluation Report</title></head><body>\n")
            f.write("<h1>Evaluation Report</h1>\n")
            f.write("<table border=\"1\" style=\"border-collapse:collapse;\">\n")
            f.write("<tr><th>Image</th><th>Name</th><th>Metadata</th></tr>\n")
            for r in self.rows:
                f.write("<tr>")
                f.write(f"<td><img src=\"{r['image_path']}\" width=300></td>")
                f.write(f"<td>{r['name']}</td>")
                f.write(f"<td><pre>{r['metadata']}</pre></td>")
                f.write("</tr>\n")
            f.write("</table>\n")
            f.write("</body></html>")
