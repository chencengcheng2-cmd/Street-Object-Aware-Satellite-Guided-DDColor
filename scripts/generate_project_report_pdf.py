from pathlib import Path
import json
import math
import shutil
from PIL import Image, ImageDraw, ImageFont

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

PROJECT = Path(r'C:\Users\31133\Desktop\satellite_guided_ddcolor')
OUT_DIR = PROJECT / 'output' / 'pdf'
ASSET_DIR = OUT_DIR / 'assets'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / 'satellite_guided_ddcolor_project_report_zh.pdf'
DESKTOP_COPY = Path(r'C:\Users\31133\Desktop\satellite_guided_ddcolor_中文项目报告.pdf')

METRICS_PATH = Path(r'C:\Users\31133\Desktop\对比图03\metrics_summary.json')
LPIPS_FID_PATH = Path(r'C:\Users\31133\Desktop\对比图03\lpips_fid_summary.json')
with METRICS_PATH.open('r', encoding='utf-8') as f:
    metrics = json.load(f)
with LPIPS_FID_PATH.open('r', encoding='utf-8') as f:
    perceptual = json.load(f)

# Font registration.
def register_fonts():
    candidates = [
        (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\msyhbd.ttc'),
        (r'C:\Windows\Fonts\simsun.ttc', r'C:\Windows\Fonts\simhei.ttf'),
    ]
    for regular, bold in candidates:
        try:
            pdfmetrics.registerFont(TTFont('CN', regular))
            if Path(bold).exists():
                pdfmetrics.registerFont(TTFont('CN-Bold', bold))
            else:
                pdfmetrics.registerFont(TTFont('CN-Bold', regular))
            return 'CN', 'CN-Bold'
        except Exception:
            continue
    return 'Helvetica', 'Helvetica-Bold'

FONT, FONT_BOLD = register_fonts()

# PIL font.
def pil_font(size=24, bold=False):
    paths = []
    if bold:
        paths += [r'C:\Windows\Fonts\msyhbd.ttc', r'C:\Windows\Fonts\simhei.ttf']
    paths += [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simsun.ttc', r'C:\Windows\Fonts\arial.ttf']
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

# Visual assets.
def draw_round_rect(draw, box, radius, fill, outline=None, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def center_text(draw, box, text, font, fill=(20, 20, 20), spacing=4):
    lines = text.split('\n')
    line_heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    total_h = sum(line_heights) + spacing * (len(lines) - 1)
    y = box[1] + (box[3] - box[1] - total_h) / 2
    for line, w, h in zip(lines, widths, line_heights):
        x = box[0] + (box[2] - box[0] - w) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += h + spacing

def arrow(draw, start, end, fill=(45, 45, 45), width=4):
    draw.line([start, end], fill=fill, width=width)
    ang = math.atan2(end[1]-start[1], end[0]-start[0])
    length = 14
    for a in [ang + math.pi*0.82, ang - math.pi*0.82]:
        p = (end[0] + length*math.cos(a), end[1] + length*math.sin(a))
        draw.line([end, p], fill=fill, width=width)

def make_project_flow(path):
    U = lambda s: s.encode('ascii').decode('unicode_escape')
    W, H = 1800, 1050
    img = Image.new('RGB', (W, H), (248, 250, 252))
    d = ImageDraw.Draw(img)
    title_f = pil_font(44, True)
    f = pil_font(25, True)
    small = pil_font(20)
    d.text((60, 40), U('\\u9879\\u76ee\\u603b\\u4f53\\u6d41\\u7a0b\\uff1a\\u51bb\\u7ed3 DDColor + \\u536b\\u661f ViT \\u8de8\\u89c6\\u89d2\\u7ec6\\u8282\\u4fee\\u6b63'), font=title_f, fill=(15, 23, 42))

    boxes = {
        'gray': (90, 170, 380, 280, U('\\u8857\\u666f\\u7070\\u5ea6 patch\\n256 x 256')),
        'ddcolor': (500, 170, 805, 280, U('Frozen DDColor\\n\\u57fa\\u7840\\u4e0a\\u8272\\u5668')),
        'base': (930, 170, 1220, 280, U('base_rgb\\n\\u57fa\\u7840\\u5f69\\u8272\\u7ed3\\u679c')),
        'streetvit': (500, 400, 805, 525, U('Street ViT Encoder\\n\\u7070\\u5ea6 + base_rgb')),
        'sat': (90, 650, 380, 760, U('\\u536b\\u661f\\u56fe\\n256 x 256')),
        'satvit': (500, 650, 805, 760, U('Satellite ViT Encoder\\n\\u536b\\u661f token')),
        'attn': (930, 500, 1275, 680, U('Cross-View Attention\\n\\u8857\\u666f token \\u67e5\\u8be2\\u536b\\u661f token\\n+ no-match token')),
        'corr': (1430, 390, 1710, 540, U('Spatial Residual\\nCorrection')),
        'final': (1430, 710, 1710, 840, U('final_rgb\\n\\u6700\\u7ec8\\u4e0a\\u8272\\u7ed3\\u679c')),
    }
    colors_map = {
        'gray': (219, 234, 254), 'ddcolor': (224, 242, 254), 'base': (220, 252, 231),
        'streetvit': (254, 243, 199), 'sat': (236, 253, 245), 'satvit': (209, 250, 229),
        'attn': (255, 237, 213), 'corr': (237, 233, 254), 'final': (254, 226, 226)
    }
    for key, (x1, y1, x2, y2, text) in boxes.items():
        draw_round_rect(d, (x1, y1, x2, y2), 22, colors_map[key], outline=(71, 85, 105), width=3)
        center_text(d, (x1, y1, x2, y2), text, f)

    # Main DDColor branch.
    arrow(d, (380, 225), (500, 225))
    arrow(d, (805, 225), (930, 225))

    # Street ViT branch.
    arrow(d, (652, 280), (652, 400))

    # Satellite branch.
    arrow(d, (380, 705), (500, 705))

    # Cross-view fusion.
    arrow(d, (805, 462), (930, 560))
    arrow(d, (805, 705), (930, 625))

    # Base RGB conditions correction through an outer route.
    d.line([(1075, 280), (1075, 360), (1500, 360), (1500, 390)], fill=(45, 45, 45), width=4)
    arrow(d, (1500, 360), (1500, 390))

    # Attention to correction.
    arrow(d, (1275, 590), (1430, 465))

    # Correction to final.
    arrow(d, (1570, 540), (1570, 710))

    d.text((90, 900), U('\\u6838\\u5fc3\\u601d\\u60f3\\uff1aDDColor \\u5148\\u7ed9\\u51fa\\u7a33\\u5b9a\\u57fa\\u7840\\u989c\\u8272\\uff1bViT \\u53ea\\u5b66\\u4e60\\u536b\\u661f\\u4e0a\\u4e0b\\u6587\\u5bf9 DDColor \\u7ed3\\u679c\\u7684\\u6b8b\\u5dee\\u4fee\\u6b63\\uff0c\\u4e0d\\u91cd\\u65b0\\u751f\\u6210\\u6574\\u5f20\\u5f69\\u8272\\u56fe\\u3002'), font=small, fill=(51, 65, 85))
    img.save(path, quality=95)

def make_vit_flow(path):
    W, H = 1600, 1050
    img = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    title = pil_font(44, True)
    f = pil_font(25, True)
    small = pil_font(19)
    d.text((60, 36), 'No-Polar Cross-View ViT 细节流程', font=title, fill=(17, 24, 39))
    boxes = [
        ((70,150,360,270), '街景 patch\n256 x 256'),
        ((410,150,700,270), '切成 token\n32 x 32 个\n每个 8 x 8'),
        ((750,150,1040,270), 'Street Transformer\n加入上下文'),
        ((70,430,360,550), '卫星图\n256 x 256'),
        ((410,430,700,550), '切成 token\n32 x 32 个\n每个 8 x 8'),
        ((750,430,1040,550), 'Satellite Transformer\n卫星特征'),
        ((1120,290,1490,440), 'Cross Attention\n街景 token 查询所有卫星 token\n+ no-match token'),
        ((1120,560,1490,700), '生成空间特征\nfeatures + color_prior'),
        ((470,760,850,900), 'base_rgb + features\n+ context vector'),
        ((920,760,1290,900), '残差修正\ndelta_color'),
        ((1350,760,1540,900), 'final_rgb'),
    ]
    fills = [(239,246,255),(219,234,254),(191,219,254),(236,253,245),(209,250,229),(167,243,208),(255,247,237),(255,237,213),(245,243,255),(237,233,254),(254,226,226)]
    for (box, text), fill in zip(boxes, fills):
        draw_round_rect(d, box, 22, fill, outline=(71,85,105), width=3)
        center_text(d, box, text, f)
    arrow(d, (360,210), (410,210)); arrow(d, (700,210), (750,210))
    arrow(d, (360,490), (410,490)); arrow(d, (700,490), (750,490))
    arrow(d, (1040,210), (1120,335)); arrow(d, (1040,490), (1120,395))
    arrow(d, (1305,440), (1305,560)); arrow(d, (1305,700), (1040,820)); arrow(d, (850,830), (920,830)); arrow(d, (1290,830), (1350,830))
    d.text((70, 950), '注意：attention 找的是深层特征相关 token，不是直接复制卫星图颜色。天空等无对应区域可通过 no-match token 避免强行匹配。', font=small, fill=(55,65,81))
    img.save(path, quality=95)

def make_metric_chart(path):
    W, H = 1500, 760
    img = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    title = pil_font(40, True)
    f = pil_font(23)
    bf = pil_font(25, True)
    d.text((60, 40), '500 张测试集指标对比：DDColor Base vs No-Polar Cross-View ViT', font=title, fill=(17,24,39))
    items = [
        ('PSNR ↑', metrics['base_psnr'], metrics['final_psnr'], 35.0),
        ('SSIM ↑', metrics['base_ssim'], metrics['final_ssim'], 1.0),
        ('LPIPS ↓', perceptual['base_lpips'], perceptual['final_lpips'], 0.20),
        ('FID ↓', perceptual['base_fid'], perceptual['final_fid'], 20.0),
        ('MAE ↓', metrics['base_mae'], metrics['final_mae'], 0.06),
        ('RMSE ↓', metrics['base_rmse'], metrics['final_rmse'], 0.08),
    ]
    x0, y0 = 230, 155
    group_gap = 195
    bar_w = 50
    max_h = 390
    d.line((120, y0+max_h, 1430, y0+max_h), fill=(148,163,184), width=3)
    for i, (name, base, final, ymax) in enumerate(items):
        gx = x0 + i * group_gap
        bh = base / ymax * max_h
        fh = final / ymax * max_h
        d.rounded_rectangle((gx, y0+max_h-bh, gx+bar_w, y0+max_h), radius=8, fill=(148,163,184))
        d.rounded_rectangle((gx+bar_w+18, y0+max_h-fh, gx+bar_w*2+18, y0+max_h), radius=8, fill=(37,99,235))
        d.text((gx-15, y0+max_h+18), name, font=bf, fill=(15,23,42))
        d.text((gx-30, y0+max_h-bh-34), f'{base:.3f}', font=f, fill=(71,85,105))
        d.text((gx+bar_w+2, y0+max_h-fh-34), f'{final:.3f}', font=f, fill=(30,64,175))
    d.rounded_rectangle((1030, 80, 1430, 135), radius=14, fill=(248,250,252), outline=(203,213,225), width=2)
    d.rectangle((1052, 100, 1085, 120), fill=(148,163,184)); d.text((1095, 94), 'DDColor Base', font=f, fill=(31,41,55))
    d.rectangle((1255, 100, 1288, 120), fill=(37,99,235)); d.text((1298, 94), 'Final', font=f, fill=(31,41,55))
    d.text((60, 690), f"PSNR +{metrics['psnr_improvement']:.2f} dB, LPIPS 降低 {perceptual['lpips_reduction_percent']:.2f}%, FID 降低 {perceptual['fid_reduction_percent']:.2f}%", font=bf, fill=(22,101,52))
    img.save(path, quality=95)

project_flow = ASSET_DIR / 'project_flow.jpg'
vit_flow = ASSET_DIR / 'vit_flow.jpg'
metric_chart = ASSET_DIR / 'metric_chart.jpg'
make_project_flow(project_flow)
make_vit_flow(vit_flow)
make_metric_chart(metric_chart)

# Styles.
styles = getSampleStyleSheet()
for s in styles.byName.values():
    s.fontName = FONT
    s.wordWrap = 'CJK'
styles.add(ParagraphStyle('CNTitle', parent=styles['Title'], fontName=FONT_BOLD, fontSize=26, leading=34, alignment=TA_CENTER, spaceAfter=16, wordWrap='CJK'))
styles.add(ParagraphStyle('CNHeading1', parent=styles['Heading1'], fontName=FONT_BOLD, fontSize=18, leading=24, spaceBefore=16, spaceAfter=10, wordWrap='CJK'))
styles.add(ParagraphStyle('CNHeading2', parent=styles['Heading2'], fontName=FONT_BOLD, fontSize=14, leading=20, spaceBefore=10, spaceAfter=6, wordWrap='CJK'))
styles.add(ParagraphStyle('CNBody', parent=styles['BodyText'], fontName=FONT, fontSize=10.5, leading=17, spaceAfter=6, wordWrap='CJK'))
styles.add(ParagraphStyle('CNNote', parent=styles['BodyText'], fontName=FONT, fontSize=9, leading=14, textColor=colors.HexColor('#475569'), wordWrap='CJK'))
styles.add(ParagraphStyle('Caption', parent=styles['BodyText'], fontName=FONT, fontSize=9, leading=13, alignment=TA_CENTER, textColor=colors.HexColor('#475569'), spaceAfter=8, wordWrap='CJK'))

PAGE_W, PAGE_H = A4
MARGIN = 17 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

def P(text, style='CNBody'):
    return Paragraph(text, styles[style])

def image_flow(path, max_w=CONTENT_W, max_h=170*mm, caption=None):
    im = Image.open(path)
    w, h = im.size
    scale = min(max_w / w, max_h / h)
    flow = [RLImage(str(path), width=w*scale, height=h*scale)]
    if caption:
        flow.append(P(caption, 'Caption'))
    return flow

def make_table(data, col_widths=None, font_size=8.5):
    cell_style = ParagraphStyle(
        'TableCell',
        fontName=FONT,
        fontSize=font_size,
        leading=font_size + 4,
        wordWrap='CJK',
        splitLongWords=True,
        spaceAfter=0,
    )
    header_style = ParagraphStyle(
        'TableHeader',
        parent=cell_style,
        fontName=FONT_BOLD,
        textColor=colors.HexColor('#0F172A'),
    )

    wrapped = []
    for row_i, row in enumerate(data):
        style = header_style if row_i == 0 else cell_style
        wrapped.append([Paragraph(str(cell).replace(' + ', ' +<br/>'), style) for cell in row])

    table = Table(wrapped, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('GRID', (0,0), (-1,-1), 0.45, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (1,1), (-1,-1), 'LEFT'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    return table

story = []
story.append(P('卫星引导 DDColor 街景灰度图像上色项目报告', 'CNTitle'))
story.append(P('Frozen DDColor + Cross-View ViT + Residual Color Correction', 'Caption'))
story.append(Spacer(1, 8))
story.append(P('报告日期：2026 年 6 月 4 日', 'CNBody'))
story.append(P('项目路径：C:/Users/31133/Desktop/satellite_guided_ddcolor', 'CNBody'))
story.append(P('当前主模型：cross_view_vit_no_polar_v4，权重文件：checkpoints/cross_view_vit_no_polar_v4/best.pth', 'CNBody'))
story.append(Spacer(1, 10))
story += image_flow(project_flow, max_h=130*mm, caption='图 1：当前项目总体流程。')
story.append(PageBreak())

story.append(P('1. 项目目标', 'CNHeading1'))
story.append(P('本项目的目标是在不重新训练 DDColor 主体模型的前提下，利用卫星图提供的跨视角上下文信息，对 DDColor 生成的街景基础上色结果进行细节颜色修正。重点关注道路、路面、标线等局部细节，而不是简单做整图全局调色。', 'CNBody'))
story.append(P('核心约束是：DDColor 完全冻结，只作为基础上色器；可训练部分只学习 residual correction，即学习 delta_color，再与 DDColor 的 base_rgb 相加得到最终结果。', 'CNBody'))
story.append(P('最终形式为：final_rgb = clamp(base_rgb + delta_color, 0, 1)。这样模型不需要从零生成彩色图，而是学习如何修正 DDColor 的错误颜色。', 'CNBody'))

story.append(P('2. 数据与实验设置', 'CNHeading1'))
data_table = [
    ['项目', '设置'],
    ['训练数据', 'CVUSA_processed_split，训练样本约 15284，验证样本 576'],
    ['测试数据', 'C:/Users/31133/Desktop/polar and bing，取 500 组样本'],
    ['街景输入', '全景图 resize 到 1024 x 256，再切成 4 个 256 x 256 patch'],
    ['卫星输入', '完整卫星图 resize 到 256 x 256，不切分'],
    ['Polar 图', '当前 no-polar v4 不使用 Polar；旧实验中使用过 Polar'],
    ['训练参数', 'batch size 4，epoch 30，AdamW，lr 4e-5，AMP mixed precision'],
    ['训练耗时', '7.94 小时'],
    ['验证最好结果', 'Best PSNR = 32.15；最后一轮验证 PSNR = 32.07，SSIM = 0.9867'],
]
story.append(make_table(data_table, col_widths=[35*mm, CONTENT_W-35*mm], font_size=9))

story.append(PageBreak())
story.append(P('3. 模型演进与实验对比', 'CNHeading1'))
story.append(P('项目经历了三个主要版本：第一版使用 Polar 全局上下文 + FiLM；第二版使用卫星图、Polar 图和方向编码的 dual-context；第三版转向 no-polar Cross-View ViT，直接用街景 token 和卫星 token 建立弱对应关系。', 'CNBody'))
exp_table = [
    ['版本', '主要输入', '核心结构', '说明'],
    ['Polar-FiLM', '灰度街景 + Polar', 'Polar Encoder + FiLM + Residual', '第一版，主要做全局上下文调色'],
    ['Dual-Context', '灰度街景 + 卫星 + Polar + 方向', 'Satellite Encoder + Polar Encoder + Direction Fusion', '比第一版略有提升，但仍偏全局调色'],
    ['No-Polar ViT', '灰度街景 + 卫星 + 方向', 'Street ViT + Satellite ViT + Cross Attention + Residual', '当前主版本，重点尝试跨视角 token 细节对应'],
]
story.append(make_table(exp_table, col_widths=[34*mm, 42*mm, 66*mm, CONTENT_W-142*mm], font_size=8))
story.append(P('完整实验名：Polar-FiLM = film_ddcolor_cu130_20260527；Dual-Context = dual_context_v1；No-Polar ViT = cross_view_vit_no_polar_v4。', 'CNNote'))
story.append(PageBreak())

story.append(P('4. 当前模型总体流程', 'CNHeading1'))
story.append(P('当前 no-polar v4 版本不再使用 Polar 图参与模型计算。它使用街景灰度 patch、DDColor base_rgb、卫星图和方向编号。街景 patch 与卫星图分别经过 ViT 编码后，通过 Cross-View Attention 做软对应，再将得到的空间特征和颜色先验输入残差修正模块。', 'CNBody'))
story += image_flow(project_flow, max_h=135*mm, caption='图 2：当前模型从输入到最终输出的完整流程。')
story.append(PageBreak())

story.append(P('5. ViT 跨视角流程重点说明', 'CNHeading1'))
story.append(P('ViT 的关键作用不是直接复制卫星图颜色，而是将街景和卫星图切成 token，分别提取特征，再通过 attention 计算街景 token 对卫星 token 的关注权重。当前 patch_size 为 8，因此每张 256 x 256 图像会得到 32 x 32 = 1024 个 token。', 'CNBody'))
story += image_flow(vit_flow, max_h=150*mm, caption='图 3：No-Polar Cross-View ViT 细节流程。')
story.append(P('对于每个街景 token，模型会查询所有卫星 token。attention 权重较高的位置代表模型认为这些卫星 token 与当前街景 token 的深层特征更相关。由于街景和卫星图视角差异极大，模型还加入 no-match token，用于表示天空等卫星图中不存在的区域。', 'CNBody'))
story.append(P('当前观察到的问题是：很多道路 token 的 attention 并没有稳定集中到卫星道路区域。这说明模型确实学习了跨视角 token 对应，但对应关系仍不够可靠，特别是细小标线区域。', 'CNBody'))
story.append(PageBreak())

story.append(P('6. Token 大小与对应关系可视化', 'CNHeading1'))
token_grid = PROJECT / 'outputs' / 'token_visualization' / '0001255_token_grid_explanation_better.jpg'
story += image_flow(token_grid, max_h=150*mm, caption='图 4：街景 patch 和卫星图的 token 网格。一个 token 对应 8 x 8 像素。')
story.append(P('该图展示了完整街景图如何切成 4 个 patch，以及单个街景 patch 与卫星图如何切成 8 x 8 token。当前模型中，单个街景 patch 有 1024 个 street tokens，卫星图也有 1024 个 satellite tokens。', 'CNBody'))
story.append(PageBreak())

story.append(P('7. Token Attention 对应图', 'CNHeading1'))
attention_single = PROJECT / 'outputs' / 'token_visualization' / '0003606_road_token_attention_best.jpg'
story += image_flow(attention_single, max_h=160*mm, caption='图 5：马路 token 对卫星图 token 的 attention 热力图。')
story.append(P('左侧红框表示选中的街景道路 token，右侧热力图表示该 token 对卫星图各 token 的 attention 权重。红色或黄色区域表示关注程度高，蓝色区域表示关注程度低。编号框为权重最高的若干卫星 token。', 'CNBody'))
story.append(P('这类图用于判断 Cross-View Attention 是否真正把街景道路 token 对应到卫星图道路区域。如果热力图分散或落在建筑/植被区域，说明跨视角对应不稳定。', 'CNBody'))
story.append(PageBreak())

story.append(P('8. 多样本 Road Token Attention 总览', 'CNHeading1'))
attention_batch = PROJECT / 'outputs' / 'token_visualization' / 'road_token_batch' / 'road_token_attention_batch_contact_sheet.jpg'
story += image_flow(attention_batch, max_h=235*mm, caption='图 6：多样本马路 token 对应关系总览。')
story.append(PageBreak())

story.append(P('9. 效果指标提升', 'CNHeading1'))
story.append(P('当前 no-polar Cross-View ViT 在 500 张测试样本上，相比 DDColor base 输出有明显指标提升。PSNR、SSIM 越高越好；LPIPS、FID、MAE、RMSE 越低越好。', 'CNBody'))
story += image_flow(metric_chart, max_h=120*mm, caption='图 7：500 张测试样本的主要指标对比。')
metric_table = [
    ['指标', 'DDColor Base', 'No-Polar ViT Final', '变化', '方向'],
    ['PSNR', f"{metrics['base_psnr']:.4f}", f"{metrics['final_psnr']:.4f}", f"+{metrics['psnr_improvement']:.4f}", '越高越好'],
    ['SSIM', f"{metrics['base_ssim']:.4f}", f"{metrics['final_ssim']:.4f}", f"+{metrics['ssim_improvement']:.4f}", '越高越好'],
    ['LPIPS', f"{perceptual['base_lpips']:.4f}", f"{perceptual['final_lpips']:.4f}", f"-{perceptual['lpips_reduction']:.4f}", '越低越好'],
    ['FID', f"{perceptual['base_fid']:.4f}", f"{perceptual['final_fid']:.4f}", f"-{perceptual['fid_reduction']:.4f}", '越低越好'],
    ['MAE', f"{metrics['base_mae']:.5f}", f"{metrics['final_mae']:.5f}", f"-{metrics['mae_reduction']:.5f}", '越低越好'],
    ['RMSE', f"{metrics['base_rmse']:.5f}", f"{metrics['final_rmse']:.5f}", f"-{metrics['rmse_reduction']:.5f}", '越低越好'],
]
story.append(make_table(metric_table, col_widths=[24*mm, 34*mm, 38*mm, 30*mm, 30*mm], font_size=8.5))
story.append(P(f"MAE 降低 {metrics['mae_reduction_percent']:.2f}%，RMSE 降低 {metrics['rmse_reduction_percent']:.2f}%，LPIPS 降低 {perceptual['lpips_reduction_percent']:.2f}%，FID 降低 {perceptual['fid_reduction_percent']:.2f}%。", 'CNBody'))
story.append(PageBreak())

story.append(P('10. 不同实验版本的横向对比', 'CNHeading1'))
story.append(P('下表汇总了在同一 500 张测试集上的主要版本结果。第一版与 dual-context v1 的 LPIPS/FID 为此前实验记录；当前 no-polar ViT 已补算 LPIPS/FID。', 'CNBody'))
compare_table = [
    ['模型/结果', 'PSNR', 'SSIM', 'LPIPS', 'FID', 'MAE', 'RMSE'],
    ['DDColor Base', '24.1411', '0.9549', '0.1665', '17.5780', '0.04514', '0.06551'],
    ['第一版 Polar-FiLM Residual', '29.1311', '0.9782', '0.1215', '17.4300', '0.02508', '0.03646'],
    ['dual_context_v1', '29.3342', '0.9784', '0.1188', '16.6259', '0.02426', '0.03574'],
    ['当前 No-Polar Cross-View ViT', f"{metrics['final_psnr']:.4f}", f"{metrics['final_ssim']:.4f}", f"{perceptual['final_lpips']:.4f}", f"{perceptual['final_fid']:.4f}", f"{metrics['final_mae']:.5f}", f"{metrics['final_rmse']:.5f}"],
]
story.append(make_table(compare_table, col_widths=[48*mm, 22*mm, 22*mm, 22*mm, 22*mm, 24*mm, 24*mm], font_size=7.8))
story.append(P('从整体指标看，当前 No-Polar Cross-View ViT 明显优于前两个版本，尤其是 PSNR、LPIPS 和 MAE。但从 token attention 可视化看，细粒度跨视角对应仍不稳定，因此视觉上对黄色标线等局部结构的提升并不总是稳定。', 'CNBody'))
story.append(PageBreak())

C = lambda s: s.encode('ascii').decode('unicode_escape')
story.append(P(C(r'11. \u6548\u679c\u5bf9\u6bd4\u56fe'), 'CNHeading1'))
story.append(P(C(r'\u672c\u8282\u589e\u52a0\u66f4\u591a\u6d4b\u8bd5\u6837\u672c\u7684\u53ef\u89c6\u5316\u5bf9\u6bd4\u3002\u6bcf\u5f20\u56fe\u4ece\u5de6\u5230\u53f3\u5206\u522b\u4e3a DDColor Base\u3001Improved Final \u548c Streetview GT\u3002'), 'CNBody'))
compare_dir = Path.home() / 'Desktop' / C(r'\u5bf9\u6bd4\u56fe03')
preferred_ids = ['0000008','0000029','0000035','0000052','0000068','0000075','0000082','0000094','0000208','0000335','0000544','0000846']
compare_imgs = [compare_dir / f'{sid}_comparison.jpg' for sid in preferred_ids]
if not any(p.exists() for p in compare_imgs):
    compare_imgs = sorted(compare_dir.glob('*_comparison.jpg'))[:12]
for page_i in range(0, len(compare_imgs), 4):
    group = compare_imgs[page_i:page_i + 4]
    for local_i, img_path in enumerate(group, start=1):
        if img_path.exists():
            fig_no = page_i + local_i
            story += image_flow(img_path, max_h=48*mm, caption=C(r'\u56fe 8.' + str(fig_no) + r'\uff1aDDColor Base | Improved Final | Streetview GT\u3002'))
    if page_i + 4 < len(compare_imgs):
        story.append(PageBreak())
        story.append(P(C(r'11. \u6548\u679c\u5bf9\u6bd4\u56fe\uff08\u7eed\uff09'), 'CNHeading1'))
story.append(P(C(r'\u4ece\u66f4\u591a\u6837\u672c\u53ef\u4ee5\u770b\u5230\uff0c\u5f53\u524d\u6a21\u578b\u5728\u6574\u4f53\u989c\u8272\u3001\u8def\u9762\u8272\u8c03\u3001\u690d\u88ab\u548c\u5929\u7a7a\u4e00\u81f4\u6027\u4e0a\u901a\u5e38\u4f18\u4e8e DDColor Base\uff1b\u4f46\u6781\u7ec6\u7684\u8f66\u9053\u7ebf\u3001\u9ec4\u8272\u6807\u7ebf\u548c\u5c40\u90e8\u9053\u8def\u7eb9\u7406\u4ecd\u53ef\u80fd\u4e0d\u7a33\u5b9a\u3002'), 'CNBody'))
story.append(PageBreak())


story.append(P('12. 当前问题与技术判断', 'CNHeading1'))
issues = [
    ['问题', '原因', '影响'],
    ['很多 token 对应错误', '街景为透视视角，卫星为俯视视角，没有人工 token 对应标签', 'attention 容易分散或落到错误区域'],
    ['8 x 8 token 太小', '单个 token 只包含很少局部纹理，缺少语义上下文', '道路/建筑/阴影/标线容易混淆'],
    ['黄色标线不稳定', '标线占图像比例很小，普通 L1/Perceptual loss 对其惩罚不足', '整体指标提升，但局部细节可能仍不理想'],
    ['卫星颜色与街景颜色不完全一致', '光照、时间、地图瓦片处理、阴影不同', '不能简单复制卫星颜色'],
    ['天空没有卫星对应', '卫星图不存在天空区域', '需要 no-match 或区域约束避免错误修正'],
]
story.append(make_table(issues, col_widths=[36*mm, 78*mm, CONTENT_W-114*mm], font_size=7.8))
story.append(P('因此，当前结果说明 ViT 跨视角方案具有提升整体上色质量的能力，但如果目标转向道路标线等微小细节，需要进一步加入几何约束、道路区域约束或多尺度 token 设计。', 'CNBody'))

story.append(P('13. 下一步改进建议', 'CNHeading1'))
next_table = [
    ['方向', '具体做法', '目的'],
    ['多尺度 Street Encoder', '同时使用 8 x 8、16 x 16、32 x 32 token', '大 token 判断道路语义，小 token 保留标线细节'],
    ['Road-Guided Detail Correction', '先自动估计街景道路/标线 mask，只在道路区域增强', '减少天空、建筑、植被被错误修正'],
    ['Attention 约束', '限制道路 token 主要关注卫星道路候选区域', '降低错误 token 对应'],
    ['细节损失增强', '加强 lane marking loss、edge loss、下半部分 road loss', '让训练更关注黄色线、白线和路缘'],
    ['卫星高分辨率分支', '卫星图不只压到 256 x 256，可增加局部高分辨率 crop', '减少标线在卫星图中被压缩丢失'],
]
story.append(make_table(next_table, col_widths=[39*mm, 82*mm, CONTENT_W-121*mm], font_size=7.8))
story.append(P('推荐下一版从“自由 Cross-View Attention”改为“Road-Guided Multi-Scale Detail Correction”。这样更符合当前目标：不是整图全局调色，而是重点恢复道路和标线细节。', 'CNBody'))

# Footer callback.
def on_page(c: canvas.Canvas, doc):
    c.saveState()
    c.setFont(FONT, 8)
    c.setFillColor(colors.HexColor('#64748B'))
    c.drawString(MARGIN, 10*mm, 'Satellite-Guided DDColor Project Report')
    c.drawRightString(PAGE_W - MARGIN, 10*mm, f'第 {doc.page} 页')
    c.restoreState()

pdf = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=A4,
    rightMargin=MARGIN,
    leftMargin=MARGIN,
    topMargin=16*mm,
    bottomMargin=16*mm,
    title='卫星引导 DDColor 街景灰度图像上色项目报告',
    author='satellite_guided_ddcolor project',
)
pdf.build(story, onFirstPage=on_page, onLaterPages=on_page)
shutil.copy2(PDF_PATH, DESKTOP_COPY)
print(PDF_PATH)
print(DESKTOP_COPY)
