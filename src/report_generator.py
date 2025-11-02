import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self, analysis_results: Dict[str, Any]) -> Path:
        logger.info("  → Generating HTML report...")
        md = analysis_results["metadata"]
        top = analysis_results["top_movies"]
        rdist = analysis_results["rating_dist"]
        plots = analysis_results["plots"]

        def enc(p):
            return base64.b64encode(Path(p).read_bytes()).decode("utf-8")

        m_disp = top[0]["m"] if top else 0
        c_disp = top[0]["C"] if top else rdist["statistics"]["mean"]
        html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'/><title>MovieLens Report</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;background:#f6f7fb;margin:0;padding:24px;color:#222}}.c{{max-width:1100px;margin:0 auto;background:#fff;padding:24px;border-radius:10px;box-shadow:0 10px 30px rgba(0,0,0,.06)}}h2{{border-bottom:2px solid #667eea;padding-bottom:6px}}</style>
</head><body><div class='c'><h1>📊 MovieLens Data Analysis</h1><p>Generated {md['analysis_date']}</p>
<h2>Summary</h2><ul><li>Dataset: {md['dataset']}</li><li>Movies: {md['n_movies']:,}</li><li>Ratings: {md['n_ratings']:,}</li><li>Users: {md['n_users']:,}</li><li>Range: {md['date_range'][0]} → {md['date_range'][1]}</li></ul>
<h2>Rating Statistics</h2><img style='max-width:100%' src='data:image/png;base64,{enc(plots['rating_distribution'])}'/>
<h2>Top Movies (WR with m≈{m_disp}, C≈{c_disp:.3f})</h2><img style='max-width:100%' src='data:image/png;base64,{enc(plots['top_movies'])}'/>
<h2>Genres</h2><img style='max-width:100%' src='data:image/png;base64,{enc(plots['genre_popularity'])}'/><img style='max-width:100%' src='data:image/png;base64,{enc(plots['genre_trends'])}'/>
<h2>Time Series</h2><img style='max-width:100%' src='data:image/png;base64,{enc(plots['time_series'])}'/>
<h2>User Behavior</h2><img style='max-width:100%' src='data:image/png;base64,{enc(plots['user_activity'])}'/>
<h2>Additional</h2><img style='max-width:100%' src='data:image/png;base64,{enc(plots['rating_heatmap'])}'/><img style='max-width:100%' src='data:image/png;base64,{enc(plots['correlation_matrix'])}'/>
</div></body></html>"""
        path = (
            self.output_dir
            / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        path.write_text(html, encoding="utf-8")
        logger.info(f"  ✓ HTML report saved: {path.name}")
        return path

    def generate_pdf_summary(self, analysis_results: Dict[str, Any]) -> Path:
        logger.info("  → Generating PDF summary...")
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )

            md = analysis_results["metadata"]
            top = analysis_results["top_movies"]
            pdf_path = (
                self.output_dir
                / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph("MovieLens Analysis Report", styles["Title"]))
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(f"Dataset: {md['dataset']}", styles["Normal"]))
            story.append(
                Paragraph(
                    f"Movies: {md['n_movies']:,} • Ratings: {md['n_ratings']:,} • Users: {md['n_users']:,}",
                    styles["Normal"],
                )
            )
            data = [["Rank", "Title", "Avg", "Votes", "WR"]]
            for i, m in enumerate(top[:10], 1):
                title = m["title"][:40] + ("..." if len(m["title"]) > 40 else "")
                data.append(
                    [
                        str(i),
                        title,
                        f"{m['rating_mean']:.2f}",
                        f"{m['vote_count']:,}",
                        f"{m['weighted_rating']:.2f}",
                    ]
                )
            table = Table(data)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ]
                )
            )
            story.append(table)
            doc.build(story)
            return pdf_path
        except Exception:
            t = self.output_dir / "pdf_generation_skipped.txt"
            t.write_text(
                "Install reportlab to enable PDF output: pip install reportlab"
            )
            return t
