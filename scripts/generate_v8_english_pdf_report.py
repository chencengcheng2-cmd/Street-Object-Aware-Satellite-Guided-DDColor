"""Generate an English PDF report for the current v8 satellite-context model."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "output" / "pdf"
ASSET_DIR = OUT_DIR / "v8_english_report_assets"
PDF_PATH = OUT_DIR / "satellite_guided_ddcolor_v8_technical_report_en.pdf"
DESKTOP_PDF = Path(r"C:\Users\31133\Desktop\satellite_guided_ddcolor_v8_technical_report_en.pdf")
EXAMPLE_DIR = Path(r"C:\Users\31133\Desktop\v8_satellite_report_5_examples")
DATA_ROOT = Path(r"C:\Users\31133\Desktop\polar and bing")
TOKEN_DIR = PROJECT / "outputs" / "token_visualization"


def font_path() -> str | None:
    for path in [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]:
        if Path(path).exists():
            return path
    return None


FONT_PATH = font_path()
if FONT_PATH:
    pdfmetrics.registerFont(TTFont("ReportFont", FONT_PATH))
    BASE_FONT = "ReportFont"
else:
    BASE_FONT = "Helvetica"


def pil_font(size: int):
    if FONT_PATH:
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="ReportTitle", fontName=BASE_FONT, fontSize=25, leading=31,
        alignment=TA_CENTER, textColor=colors.HexColor("#0f172a"), spaceAfter=8))
    base.add(ParagraphStyle(
        name="SubTitle", fontName=BASE_FONT, fontSize=12.5, leading=17,
        alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceAfter=12))
    base.add(ParagraphStyle(
        name="H1", fontName=BASE_FONT, fontSize=17, leading=22,
        textColor=colors.HexColor("#0f172a"), spaceAfter=7))
    base.add(ParagraphStyle(
        name="Body", fontName=BASE_FONT, fontSize=10, leading=14.5,
        textColor=colors.HexColor("#1e293b"), alignment=TA_JUSTIFY, spaceAfter=4))
    base.add(ParagraphStyle(
        name="Caption", fontName=BASE_FONT, fontSize=8.5, leading=11,
        alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceAfter=4))
    base.add(ParagraphStyle(
        name="Cell", fontName=BASE_FONT, fontSize=8.2, leading=10.5,
        textColor=colors.HexColor("#1e293b")))
    base.add(ParagraphStyle(
        name="HeadCell", fontName=BASE_FONT, fontSize=8.5, leading=10.5,
        textColor=colors.HexColor("#0f172a")))
    return base


STYLES = styles()


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, fill=(51, 65, 85), width=5):
    draw.line([start, end], fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    size = 18
    pts = [
        (ex, ey),
        (ex - size * math.cos(angle - 0.45), ey - size * math.sin(angle - 0.45)),
        (ex - size * math.cos(angle + 0.45), ey - size * math.sin(angle + 0.45)),
    ]
    draw.polygon(pts, fill=fill)


def draw_box(draw: ImageDraw.ImageDraw, box, text: str, fill):
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=(148, 163, 184), width=3)
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    font = pil_font(26)
    total = len(lines) * 33
    y = y0 + (y1 - y0 - total) / 2
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        draw.text((x0 + (x1 - x0 - (bb[2] - bb[0])) / 2, y), line, font=font, fill="white")
        y += 33


def make_architecture(path: Path):
    img = Image.new("RGB", (1650, 940), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.text((55, 35), "Current v8 Pipeline: Frozen DDColor + Satellite Semantic Token Matching",
              font=pil_font(42), fill=(15, 23, 42))
    draw.text((55, 90), "No Polar input. The context image is the raw satellite image.",
              font=pil_font(24), fill=(71, 85, 105))
    boxes = [
        ((70, 180, 340, 300), "Grayscale street\npatch 256 x 256", (30, 64, 175)),
        ((430, 180, 700, 300), "Frozen DDColor\nbase colorizer", (14, 116, 144)),
        ((790, 180, 1060, 300), "base_rgb\ninitial color", (22, 101, 52)),
        ((430, 390, 700, 510), "Street semantic\nmask", (124, 45, 18)),
        ((790, 390, 1060, 510), "Street tokens\n16 x 16 pixels", (88, 28, 135)),
        ((70, 640, 340, 760), "Raw satellite\nimage", (15, 118, 110)),
        ((430, 640, 700, 760), "Satellite semantic\nmask", (12, 74, 110)),
        ((790, 640, 1060, 760), "Satellite tokens\n8 x 8 pixels", (21, 94, 117)),
        ((1170, 390, 1520, 575), "Semantic + color-aware\ncross-view attention\n+ no-match token", (146, 64, 14)),
        ((1170, 690, 1520, 810), "Residual correction\nfinal_rgb", (127, 29, 29)),
    ]
    for box, text, fill in boxes:
        draw_box(draw, box, text, fill)
    arrows = [
        ((340, 240), (430, 240)), ((700, 240), (790, 240)), ((925, 300), (925, 390)),
        ((340, 240), (430, 450)), ((700, 450), (790, 450)),
        ((340, 700), (430, 700)), ((700, 700), (790, 700)),
        ((1060, 450), (1170, 475)), ((1060, 700), (1170, 520)), ((1345, 575), (1345, 690)),
        ((1060, 240), (1220, 690)),
    ]
    for start, end in arrows:
        draw_arrow(draw, start, end)
    draw.text((70, 850), "Residual rule: final_rgb = clamp(base_rgb + delta_color, 0, 1). DDColor weights are frozen.",
              font=pil_font(24), fill=(51, 65, 85))
    img.save(path, quality=95)


def draw_grid(image: Image.Image, step: int, color, width=1):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for x in range(0, w + 1, step):
        draw.line([(x, 0), (x, h)], fill=color, width=width)
    for y in range(0, h + 1, step):
        draw.line([(0, y), (w, y)], fill=color, width=width)
    return out


def make_token_grid(path: Path, sample_id="0000008"):
    street = Image.open(DATA_ROOT / "streetview" / f"{sample_id}.jpg").convert("RGB").resize((1024, 256))
    street_patch = street.crop((0, 0, 256, 256))
    satellite = ImageOps.fit(Image.open(DATA_ROOT / "bingmap" / f"input{sample_id}.png").convert("RGB"), (256, 256))
    street_grid = draw_grid(street_patch, 16, (255, 230, 0), 1).resize((430, 430), Image.Resampling.NEAREST)
    sat_grid = draw_grid(satellite, 8, (255, 255, 255), 1).resize((430, 430), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1200, 650), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 30), "Token Size in the Current v8 Model", font=pil_font(42), fill=(15, 23, 42))
    draw.text((70, 108), "Street: 16 x 16 pixels", font=pil_font(23), fill=(51, 65, 85))
    draw.text((70, 138), "16 x 16 grid = 256 tokens per patch", font=pil_font(20), fill=(71, 85, 105))
    draw.text((675, 108), "Satellite: 8 x 8 pixels", font=pil_font(23), fill=(51, 65, 85))
    draw.text((675, 138), "32 x 32 grid = 1024 candidate tokens", font=pil_font(20), fill=(71, 85, 105))
    canvas.paste(street_grid, (70, 185))
    canvas.paste(sat_grid, (675, 185))
    draw.rectangle((225, 340, 252, 367), outline=(255, 0, 0), width=5)
    draw.rectangle((900, 415, 916, 431), outline=(255, 0, 0), width=5)
    canvas.save(path, quality=95)


def make_contact_sheet(path: Path):
    files = sorted(EXAMPLE_DIR.glob("*_comparison.jpg"))[:5]
    rows = []
    for file in files:
        img = Image.open(file).convert("RGB")
        img.thumbnail((1500, 230), Image.Resampling.LANCZOS)
        rows.append((file.stem, img.copy()))
    width = 1600
    height = sum(i.height + 58 for _, i in rows) + 30
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 20
    for idx, (name, img) in enumerate(rows, 1):
        draw.text((35, y), f"Example {idx}: {name}", font=pil_font(28), fill=(15, 23, 42))
        y += 36
        canvas.paste(img, (35, y))
        y += img.height + 22
    canvas.save(path, quality=95)


def make_metric_chart(path: Path, metrics: dict):
    canvas = Image.new("RGB", (1100, 560), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((45, 30), "5-Sample Quantitative Check", font=pil_font(40), fill=(15, 23, 42))
    rows = [
        ("PSNR", metrics.get("base_psnr"), metrics.get("final_psnr"), "higher is better"),
        ("SSIM", metrics.get("base_ssim"), metrics.get("final_ssim"), "higher is better"),
        ("LPIPS", metrics.get("base_lpips"), metrics.get("final_lpips"), "lower is better"),
        ("FID", metrics.get("base_fid"), metrics.get("final_fid"), "lower; unstable at n=5"),
    ]
    for i, (name, base, final, note) in enumerate(rows):
        y = 125 + i * 95
        draw.text((55, y), name, font=pil_font(28), fill=(15, 23, 42))
        draw.text((55, y + 35), note, font=pil_font(17), fill=(100, 116, 139))
        max_v = max(float(base), float(final)) * 1.12
        draw.rounded_rectangle((250, y, 250 + int(620 * float(base) / max_v), y + 28), radius=8, fill=(148, 163, 184))
        draw.rounded_rectangle((250, y + 38, 250 + int(620 * float(final) / max_v), y + 66), radius=8, fill=(37, 99, 235))
        draw.text((900, y - 2), f"Base {float(base):.4f}", font=pil_font(22), fill=(71, 85, 105))
        draw.text((900, y + 36), f"Final {float(final):.4f}", font=pil_font(22), fill=(37, 99, 235))
    canvas.save(path, quality=95)


def add_image(story, path: Path, max_w=170 * mm, max_h=120 * mm, caption=None):
    if not path.exists():
        return
    img = Image.open(path)
    scale = min(max_w / img.width, max_h / img.height)
    story.append(PdfImage(str(path), width=img.width * scale, height=img.height * scale))
    if caption:
        story.append(Paragraph(caption, STYLES["Caption"]))
    story.append(Spacer(1, 4 * mm))


def table(data, widths):
    wrapped = []
    for r, row in enumerate(data):
        style = STYLES["HeadCell"] if r == 0 else STYLES["Cell"]
        wrapped.append([Paragraph(str(c), style) for c in row])
    t = Table(wrapped, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def fmt(value):
    return f"{float(value):.4f}" if value is not None else "N/A"


def build_pdf(metrics: dict):
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    story = []
    story.append(Paragraph("Satellite-Guided DDColor v8 Technical Report", STYLES["ReportTitle"]))
    story.append(Paragraph("Raw Satellite Semantic Token Matching for Street-View Colorization", STYLES["SubTitle"]))
    story.append(Paragraph(
        "This report explains the current v8 version after reverting from the Polar trial. The model uses the raw satellite image, not a Polar image. DDColor remains frozen and the learned modules only predict residual color correction.",
        STYLES["Body"]))
    story.append(table([
        ["Item", "Current setting"],
        ["Base model", "Frozen DDColor tiny"],
        ["Context input", "Raw satellite image resized to 256 x 256"],
        ["Semantic input", "Street semantic mask and raw satellite semantic mask"],
        ["Token sizes", "Street token 16 x 16 pixels; satellite token 8 x 8 pixels"],
        ["Correction target", "delta_color added to DDColor base output"],
    ], [45 * mm, 120 * mm]))
    story.append(PageBreak())

    story.append(Paragraph("1. Technical Pipeline", STYLES["H1"]))
    add_image(story, ASSET_DIR / "architecture.png", max_h=130 * mm, caption="Figure 1. Current v8 architecture. The context branch uses the raw satellite image and satellite semantic mask.")
    story.append(Paragraph(
        "The model first uses DDColor to produce base_rgb. Then it builds street tokens from grayscale input, base_rgb, and street semantics. It builds satellite tokens from the raw satellite RGB image and satellite semantics. Cross-view attention maps street tokens to satellite token candidates and produces a token-level color prior.",
        STYLES["Body"]))
    story.append(PageBreak())

    story.append(Paragraph("2. Token and Semantic Matching Principle", STYLES["H1"]))
    add_image(story, ASSET_DIR / "token_grid.png", max_h=105 * mm, caption="Figure 2. Street tokens are larger than satellite tokens. This balances perspective context and satellite color detail.")
    story.append(Paragraph(
        "For each street token, the matcher scores all satellite tokens. The score is biased by semantic compatibility, token color similarity, semantic distribution similarity, and boundary similarity. A no-match token is appended so that sky and unreliable regions do not need to force a satellite correspondence.",
        STYLES["Body"]))
    story.append(table([
        ["Term", "Role"],
        ["Semantic compatibility", "Encourages road-to-road, vegetation-to-vegetation, building-to-building matching."],
        ["Color bias", "Favors satellite candidates whose colors are close to the DDColor base token color."],
        ["No-match token", "Prevents sky or unmatched regions from incorrectly borrowing satellite ground colors."],
        ["Residual correction", "Learns delta_color instead of regenerating the full color image."],
    ], [45 * mm, 120 * mm]))
    story.append(PageBreak())

    story.append(Paragraph("3. Token Correspondence Visualization", STYLES["H1"]))
    attn = TOKEN_DIR / "0003606_road_token_attention_best.jpg"
    batch = TOKEN_DIR / "road_token_batch" / "road_token_attention_batch_contact_sheet.jpg"
    add_image(story, attn, max_h=120 * mm, caption="Figure 3. Example road-token attention map. The selected street token queries satellite tokens; warm colors indicate stronger attention.")
    add_image(story, batch, max_h=135 * mm, caption="Figure 4. Multiple road-token correspondence examples used to inspect whether street road regions attend to plausible satellite road regions.")
    story.append(Paragraph(
        "These visualizations are diagnostic tools. Attention is a soft learned correspondence, not a guaranteed physical location mapping. Many token correspondences can still be wrong because the street view is perspective imagery while the satellite image is overhead imagery.",
        STYLES["Body"]))
    story.append(PageBreak())

    story.append(Paragraph("4. Visual Results", STYLES["H1"]))
    story.append(Paragraph("Each row shows: grayscale input, DDColor base output, v8 final output, and street-view ground truth.", STYLES["Body"]))
    add_image(story, ASSET_DIR / "comparison_sheet.jpg", max_h=225 * mm, caption="Figure 5. Five representative comparison images generated with the current raw-satellite v8 version.")
    story.append(PageBreak())

    story.append(Paragraph("5. Quantitative Check on the 5 Examples", STYLES["H1"]))
    add_image(story, ASSET_DIR / "metrics.png", max_h=105 * mm, caption="Figure 6. Metrics on the 5 example images used in this report.")
    metric_table = [
        ["Metric", "DDColor Base", "v8 Final", "Change", "Better direction"],
        ["PSNR", fmt(metrics["base_psnr"]), fmt(metrics["final_psnr"]), f"+{metrics['psnr_improvement']:.4f}", "Higher"],
        ["SSIM", fmt(metrics["base_ssim"]), fmt(metrics["final_ssim"]), f"+{metrics['ssim_improvement']:.4f}", "Higher"],
        ["LPIPS", fmt(metrics["base_lpips"]), fmt(metrics["final_lpips"]), f"-{metrics['lpips_reduction']:.4f}", "Lower"],
        ["FID", fmt(metrics["base_fid"]), fmt(metrics["final_fid"]), f"{metrics['final_fid'] - metrics['base_fid']:.4f}", "Lower, but unreliable for only 5 images"],
    ]
    story.append(table(metric_table, [24 * mm, 32 * mm, 32 * mm, 35 * mm, 48 * mm]))
    story.append(Paragraph(
        "PSNR, SSIM, and LPIPS improve on these 5 examples. FID is worse here, but FID is not statistically reliable with only 5 images; it should be interpreted on larger evaluation sets.",
        STYLES["Body"]))
    story.append(PageBreak())

    story.append(Paragraph("6. Limitations and Next Steps", STYLES["H1"]))
    story.append(table([
        ["Limitation", "Reason", "Next step"],
        ["Lane markings are unstable", "They are small and may not dominate a token.", "Add road/lane-specific losses or a small detail branch."],
        ["Token matching is weakly supervised", "No ground-truth street-to-satellite token labels are available.", "Add geometric priors, road masks, or multi-scale matching."],
        ["Satellite and street colors differ", "Overhead imagery sees roofs, tree canopies, and lighting differently.", "Keep residual scale controlled and supervise with street RGB."],
        ["Semantic masks can be noisy", "SegFormer/NEOS classes may confuse road, building, dirt, or grass.", "Improve segmentation or use confidence-aware masks."],
    ], [38 * mm, 62 * mm, 66 * mm]))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(BASE_FONT, 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(16 * mm, 8 * mm, "Satellite-Guided DDColor v8 Technical Report")
        canvas.drawRightString(194 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((EXAMPLE_DIR / "metrics_summary.json").read_text(encoding="utf-8"))
    make_architecture(ASSET_DIR / "architecture.png")
    make_token_grid(ASSET_DIR / "token_grid.png")
    make_contact_sheet(ASSET_DIR / "comparison_sheet.jpg")
    make_metric_chart(ASSET_DIR / "metrics.png", metrics)
    build_pdf(metrics)
    shutil.copy2(PDF_PATH, DESKTOP_PDF)
    print(PDF_PATH)
    print(DESKTOP_PDF)


if __name__ == "__main__":
    main()
