import json
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

DARK_BLUE  = colors.HexColor('#1F4E5C')
LIGHT_BLUE = colors.HexColor('#EAF2F4')
SEVERE     = colors.HexColor('#C0392B')
MODERATE   = colors.HexColor('#E67E22')
MINOR      = colors.HexColor('#F1C40F')
HEALTHY    = colors.HexColor('#27AE60')
# Nothing was found, which is not the same as nothing being there. Neutral rather than
# green, and the same decision as web/css/app.css. A turbine with a blade snapped clean
# through came back from this pipeline badged green and counted as healthy, because the
# detector had found nothing it knows about. That is a fact about the detector.
NONE       = colors.HexColor('#6B7280')
LIGHT_GREY = colors.HexColor('#F5F5F5')
MID_GREY   = colors.HexColor('#CCCCCC')
W, H = A4

# Band names per subject, mirroring LABELS in web/js/severity.js. "Moderate damage" is the
# right words for a blade and the wrong ones for a crowd, which is not damaged, or a fire,
# where the finding is what is burning and who is near it.
BAND_LABELS = {
    'turbine': {'none': 'No defects found', 'minor': 'Minor wear',
                'moderate': 'Moderate damage', 'severe': 'Severe damage'},
    'solar': {'none': 'No defects found', 'minor': 'Minor wear',
              'moderate': 'Moderate damage', 'severe': 'Severe damage'},
    'crowd': {'none': 'No pressure pattern scored in this frame', 'minor': 'Worth watching',
              'moderate': 'Under pressure', 'severe': 'Needs someone now'},
    'wildfire': {'none': 'Nothing visible in this frame', 'minor': 'Minor activity',
                 'moderate': 'Active', 'severe': 'Active, with people or property'},
}

DEFAULT_BANDS = {'none': 'No findings in this image', 'minor': 'Minor',
                 'moderate': 'Moderate', 'severe': 'Severe'}

# The one-line verdict across a whole set. Mirrors summarise() in web/js/severity.js.
OVERALL_LABELS = {
    'turbine': {'severe': 'Severe damage detected', 'moderate': 'Moderate damage detected',
                'minor': 'Minor wear detected', 'none': 'Majority healthy - minor issues noted'},
    'solar': {'severe': 'Severe damage detected', 'moderate': 'Moderate damage detected',
              'minor': 'Minor wear detected', 'none': 'Majority healthy - minor issues noted'},
    'crowd': {'severe': 'Crowd pressure needing someone now',
              'moderate': 'Crowd under pressure in places',
              'minor': 'Some areas worth watching',
              'none': 'No pressure pattern scored in the frames reviewed'},
    'wildfire': {'severe': 'Active fire with people or property in the frames',
                 'moderate': 'Active fire in the frames reviewed',
                 'minor': 'Minor activity in the frames reviewed',
                 'none': 'Nothing visible in the frames reviewed'},
    None: {'severe': 'Severe findings', 'moderate': 'Moderate findings',
           'minor': 'Minor findings', 'none': 'No findings in the images reviewed'},
}

def score_to_band(score):
    """Score -> band key. Thresholds mirror web/js/severity.js; change both together."""
    if score <= 0:  return 'none'
    if score < 2:   return 'minor'
    if score < 5:   return 'moderate'
    return 'severe'

def band_of(result):
    """The band a result sits in.

    Read from `severity_band` when the exporting app recorded it, derived from the score
    otherwise. Deriving is the fallback rather than the rule because the label text now
    varies by subject, so matching on the words would break the moment one is added --
    which is exactly what happened to the gate's rejection message.
    """
    band = result.get('severity_band')
    if band in DEFAULT_BANDS:
        return band
    return score_to_band(result.get('severity_score') or 0)

def severity_colour(band):
    return {'severe': SEVERE, 'moderate': MODERATE, 'minor': MINOR}.get(band, NONE)

def score_to_label(score, domain=None):
    return BAND_LABELS.get(domain, DEFAULT_BANDS)[score_to_band(score)]

def make_styles():
    base = getSampleStyleSheet()
    def add(name, **kw):
        base.add(ParagraphStyle(name=name, **kw))
    add('SectionHead', fontSize=13, textColor=DARK_BLUE,
        fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=4)
    add('FieldValue',  fontSize=10, textColor=colors.black,
        fontName='Helvetica', spaceAfter=6)
    add('DefectItem',  fontSize=9,  textColor=colors.HexColor('#333333'),
        fontName='Helvetica', leftIndent=10, spaceAfter=2)
    add('Footer',      fontSize=7,  textColor=colors.HexColor('#AAAAAA'),
        fontName='Helvetica', alignment=TA_CENTER)
    add('Footnote',    fontSize=8,  textColor=colors.HexColor('#555555'),
        fontName='Helvetica', leading=11, spaceBefore=2)
    return base

def on_page(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.HexColor('#AAAAAA'))
    canvas.drawString(20*mm, 10*mm,
        f'Wind Turbine Inspection Report  |  Generated {datetime.now().strftime("%d %b %Y")}')
    canvas.drawRightString(W - 20*mm, 10*mm, f'Page {doc.page}')
    canvas.setStrokeColor(MID_GREY)
    canvas.setLineWidth(0.3)
    canvas.line(20*mm, 13*mm, W - 20*mm, 13*mm)
    canvas.restoreState()

def build_cover(styles, meta):
    story = []

    # header
    header_data = [
        [Paragraph(meta.get('title', 'INSPECTION REPORT'),
            ParagraphStyle('ct1', fontSize=22, textColor=colors.white,
                fontName='Helvetica-Bold', alignment=TA_CENTER, leading=30))],
        [Paragraph('Autonomous Drone Inspection System',
            ParagraphStyle('ct3', fontSize=9,
                textColor=colors.HexColor('#AACCDD'),
                fontName='Helvetica', alignment=TA_CENTER))],
    ]
    header = Table(header_data, colWidths=[W - 40*mm])
    header.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), DARK_BLUE),
        ('TOPPADDING',    (0,0),(0,0),   26),
        ('BOTTOMPADDING', (0,0),(0,0),   6),
        ('TOPPADDING',    (0,1),(0,1),   4),
        ('BOTTOMPADDING', (0,1),(0,1),   22),
        ('LEFTPADDING',   (0,0),(-1,-1), 16),
        ('RIGHTPADDING',  (0,0),(-1,-1), 16),
    ]))
    story.append(Spacer(1, 20*mm))
    story.append(header)
    story.append(Spacer(1, 10*mm))

    # meta table
    date_str = datetime.now().strftime('%d %B %Y')
    meta_rows = [
        ['Inspection Date', date_str],
        ['Asset ID',        meta.get('turbine_id', 'TRB-001')],
        ['Site',            meta.get('site', 'Wind Farm - Location TBC')],
        ['Operator',        meta.get('operator', 'DRONE EDUTRAIN LLC')],
        ['Model Used',      meta.get('model_name', 'Unspecified')],
        ['Total Images',    str(meta.get('total_images', '-'))],
    ]
    meta_tbl = Table(meta_rows, colWidths=[50*mm, 100*mm])
    meta_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(0,-1), LIGHT_BLUE),
        ('BACKGROUND',    (1,0),(1,-1), colors.white),
        ('FONTNAME',      (0,0),(0,-1), 'Helvetica-Bold'),
        ('FONTNAME',      (1,0),(1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,0),(-1,-1),10),
        ('TEXTCOLOR',     (0,0),(0,-1), DARK_BLUE),
        ('TOPPADDING',    (0,0),(-1,-1),6),
        ('BOTTOMPADDING', (0,0),(-1,-1),6),
        ('LEFTPADDING',   (0,0),(-1,-1),10),
        ('GRID',          (0,0),(-1,-1),0.3, MID_GREY),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 10*mm))

    # overall status badge
    overall = meta.get('overall_label', 'N/A')
    # The band, not the sentence. Matching on the words broke the moment the wording became
    # subject-specific, which is the same trap the gate's rejection message fell into.
    clr = severity_colour(meta.get('overall_band', 'none'))
    badge = Table([[Paragraph(f'Overall Status: {overall}',
        ParagraphStyle('badge', fontSize=14, fontName='Helvetica-Bold',
                       textColor=colors.white, alignment=TA_CENTER))]],
        colWidths=[W - 40*mm])
    badge.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), clr),
        ('TOPPADDING',    (0,0),(-1,-1), 12),
        ('BOTTOMPADDING', (0,0),(-1,-1), 12),
    ]))
    story.append(badge)
    story.append(Spacer(1, 8*mm))

    # 4 stat boxes
    s = meta.get('stats', {})
    stat_items = [
        (str(s.get('severe',   0)), 'Severe',    SEVERE),
        (str(s.get('moderate', 0)), 'Moderate',  MODERATE),
        (str(s.get('minor',    0)), 'Minor wear', MINOR),
        (str(s.get('healthy',  0)), 'Nothing marked', NONE),
    ]
    box_w = (W - 40*mm) / 4 - 3*mm
    stat_cells = []
    for num, label, c in stat_items:
        cell = Table([
            [Paragraph(num,   ParagraphStyle('sn', fontSize=26,
                fontName='Helvetica-Bold', textColor=c, alignment=TA_CENTER))],
            [Paragraph(label, ParagraphStyle('sl', fontSize=8,
                fontName='Helvetica', textColor=c, alignment=TA_CENTER))],
        ], colWidths=[box_w])
        cell.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), LIGHT_GREY),
            ('TOPPADDING',    (0,0),(-1,-1), 10),
            ('BOTTOMPADDING', (0,0),(-1,-1), 10),
            ('BOX',           (0,0),(-1,-1), 0.5, MID_GREY),
        ]))
        stat_cells.append(cell)

    stats_row = Table([stat_cells], colWidths=[(W-40*mm)/4]*4)
    stats_row.setStyle(TableStyle([
        ('LEFTPADDING',   (0,0),(-1,-1), 2),
        ('RIGHTPADDING',  (0,0),(-1,-1), 2),
        ('TOPPADDING',    (0,0),(-1,-1), 0),
        ('BOTTOMPADDING', (0,0),(-1,-1), 0),
    ]))
    story.append(stats_row)
    story.append(PageBreak())
    return story

def build_summary_table(styles, results):
    story = []
    story.append(Paragraph('Inspection Summary', styles['SectionHead']))
    story.append(HRFlowable(width='100%', thickness=1, color=DARK_BLUE, spaceAfter=6))

    headers = ['#', 'Image', 'Findings', 'People', 'Score', 'Status']
    rows = [headers]
    for i, r in enumerate(results, 1):
        dets = [d for d in r['detections'] if d['class'] != 'healthy']
        det_str = ', '.join(sorted({d['class'] for d in dets})) if dets else 'None'
        name = r['image']
        # A dash rather than a zero when nothing counted it, because "none were counted"
        # and "the count was not run" are different things and a zero says the first.
        people = r.get('people_detected')
        rows.append([
            str(i),
            name[:32] + '…' if len(name) > 32 else name,
            det_str,
            '-' if people is None else str(people),
            str(r['severity_score']),
            r['severity_label'],
        ])

    col_ws = [8*mm, 55*mm, 45*mm, 15*mm, 15*mm, 32*mm]
    tbl = Table(rows, colWidths=col_ws, repeatRows=1)
    sty = [
        ('BACKGROUND',    (0,0),(-1,0),  DARK_BLUE),
        ('TEXTCOLOR',     (0,0),(-1,0),  colors.white),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 8),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING',   (0,0),(-1,-1), 6),
        ('GRID',          (0,0),(-1,-1), 0.25, MID_GREY),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.white, LIGHT_GREY]),
        ('ALIGN',         (0,0),(0,-1),  'CENTER'),
        ('ALIGN',         (3,0),(4,-1),  'CENTER'),
    ]
    for i, r in enumerate(results, 1):
        c = severity_colour(band_of(r))
        sty.append(('TEXTCOLOR', (4,i),(4,i), c))
        sty.append(('FONTNAME',  (4,i),(4,i), 'Helvetica-Bold'))
    tbl.setStyle(TableStyle(sty))
    story.append(tbl)

    # The people footnote. Under the table rather than in a cell, because the caveat is as
    # important as the number and does not fit in fifteen millimetres.
    counted = [r for r in results if isinstance(r.get('people_detected'), int)]
    if counted:
        total = sum(r['people_detected'] for r in counted)
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            f'<b>People detected: {total}</b> across {len(counted)} '
            f'image{"" if len(counted) == 1 else "s"}. '
            'Summed per image, so anyone appearing in several is counted several times. '
            'This is what the on-device detector found, which is fewer than the people '
            'there: anyone small, distant, overlapping someone else or turned away is '
            'missed, and more are missed the higher the camera was. A floor, not a '
            'measurement.',
            styles['Footnote'],
        ))
    story.append(PageBreak())
    return story

def build_detail_pages(styles, results, annotated_dir):
    story = []
    defective = [r for r in results if r['severity_score'] > 0]
    if not defective:
        story.append(Paragraph('No defects detected in any image.', styles['FieldValue']))
        return story

    story.append(Paragraph('Defect Detail Pages', styles['SectionHead']))
    story.append(HRFlowable(width='100%', thickness=1, color=DARK_BLUE, spaceAfter=8))

    for idx, r in enumerate(defective):
        img_path = os.path.join(annotated_dir, r['image'])
        if os.path.exists(img_path):
            try:
                img = Image(img_path, width=140*mm, height=105*mm)
                img.hAlign = 'CENTER'
                story.append(img)
            except Exception:
                story.append(Paragraph(f'[Image not available: {r["image"]}]', styles['DefectItem']))
        else:
            story.append(Paragraph(f'[Image not found: {r["image"]}]', styles['DefectItem']))

        story.append(Spacer(1, 4*mm))

        clr = severity_colour(band_of(r))
        info_tbl = Table([[
            Paragraph(r['image'], ParagraphStyle('fn', fontSize=9,
                fontName='Helvetica-Bold', textColor=DARK_BLUE)),
            Paragraph(f'Score: {r["severity_score"]} / 10',
                ParagraphStyle('sc', fontSize=9, fontName='Helvetica',
                    textColor=colors.HexColor('#555555'), alignment=TA_CENTER)),
            Paragraph(r['severity_label'],
                ParagraphStyle('st', fontSize=9, fontName='Helvetica-Bold',
                    textColor=clr, alignment=TA_RIGHT)),
        ]], colWidths=[80*mm, 50*mm, 45*mm])
        info_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), LIGHT_BLUE),
            ('TOPPADDING',    (0,0),(-1,-1), 6),
            ('BOTTOMPADDING', (0,0),(-1,-1), 6),
            ('LEFTPADDING',   (0,0),(-1,-1), 8),
            ('RIGHTPADDING',  (0,0),(-1,-1), 8),
        ]))
        story.append(info_tbl)
        story.append(Spacer(1, 3*mm))

        dets = [d for d in r['detections'] if d['class'] != 'healthy']
        if dets:
            det_rows = [['Defect Type', 'Confidence', 'Bounding Box']]
            for d in dets:
                bb = d['bbox']
                det_rows.append([
                    d['class'],
                    f'{d["confidence"]*100:.1f}%',
                    f'({bb[0]:.0f}, {bb[1]:.0f}) -> ({bb[2]:.0f}, {bb[3]:.0f})'
                ])
            det_tbl = Table(det_rows, colWidths=[50*mm, 30*mm, 95*mm])
            det_tbl.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,0),  DARK_BLUE),
                ('TEXTCOLOR',     (0,0),(-1,0),  colors.white),
                ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
                ('FONTSIZE',      (0,0),(-1,-1), 8),
                ('TOPPADDING',    (0,0),(-1,-1), 4),
                ('BOTTOMPADDING', (0,0),(-1,-1), 4),
                ('LEFTPADDING',   (0,0),(-1,-1), 6),
                ('GRID',          (0,0),(-1,-1), 0.25, MID_GREY),
                ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.white, LIGHT_GREY]),
            ]))
            story.append(det_tbl)

        story.append(Spacer(1, 6*mm))
        story.append(HRFlowable(width='100%', thickness=0.3, color=MID_GREY, spaceAfter=6))
        if idx % 2 == 1:
            story.append(PageBreak())

    return story

def generate_report(
    json_path, annotated_dir, output_pdf,
    asset_id='TRB-001',
    site='Wind Farm - Location TBC',
    operator='DRONE EDUTRAIN LLC',
    model_name=None,
    title='WIND TURBINE INSPECTION REPORT',
):
    """Render an inspection PDF from an analysis JSON.

    Consumes the JSON the web app exports, so the browser tool and this generator agree on
    one schema. Results that were rejected by the domain gate or errored carry no severity
    and are excluded from the statistics - counting a refused upload as "healthy" would
    quietly inflate the pass rate, which is the one number a report must not overstate.
    """
    with open(json_path) as f:
        data = json.load(f)

    all_results = data['results']
    results = [
        r for r in all_results
        if r.get('status', 'analysed') == 'analysed' and r.get('severity_label')
    ]
    excluded = len(all_results) - len(results)

    stats = {'severe': 0, 'moderate': 0, 'minor': 0, 'healthy': 0}
    for r in results:
        band = band_of(r)
        stats['healthy' if band == 'none' else band] += 1

    total = len(results)
    if not total:
        raise SystemExit(
            f'{json_path} contains no analysed results '
            f'({excluded} rejected or errored). A report cannot be built from it, and an '
            f'empty one would read as a clean inspection of images that were never analysed.'
        )

    severe_pct   = stats['severe']   / total * 100
    moderate_pct = stats['moderate'] / total * 100
    defect_pct   = (stats['severe'] + stats['moderate'] + stats['minor']) / total * 100

    # Thresholds mirror summarise() in web/js/severity.js. Change both together, or the
    # web verdict and the PDF verdict will disagree about the same inspection.
    if severe_pct >= 10:                        overall_band = 'severe'
    elif severe_pct > 0 or moderate_pct >= 20:  overall_band = 'moderate'
    elif defect_pct >= 10:                      overall_band = 'minor'
    else:                                       overall_band = 'none'

    # One subject per report, taken from the results rather than assumed. A mixed set falls
    # back to neutral wording rather than describing a fire as damage.
    domains = {r.get('domain') for r in results if r.get('domain')}
    domain = domains.pop() if len(domains) == 1 else None
    overall = OVERALL_LABELS.get(domain, OVERALL_LABELS[None])[overall_band]

    meta = {
        'turbine_id':    asset_id,
        'site':          site,
        'operator':      operator,
        'total_images':  total,
        'overall_label': overall,
        'overall_band':  overall_band,
        'stats':         stats,
        'model_name':    model_name or data.get('model', 'Unspecified'),
        'title':         title,
    }

    styles = make_styles()
    doc = SimpleDocTemplate(
        output_pdf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm,  bottomMargin=20*mm,
    )
    story = []
    story += build_cover(styles, meta)
    story += build_summary_table(styles, results)
    story += build_detail_pages(styles, results, annotated_dir)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    print(f'Report saved to: {output_pdf}')
    print(f'  {total} images reported, {excluded} excluded (rejected or errored)')


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Render an inspection PDF from an analysis JSON '
                    '(the format exported by the web app).'
    )
    parser.add_argument('json_path', help='inspection_summary.json')
    parser.add_argument('output_pdf', help='where to write the PDF')
    parser.add_argument('--annotated-dir', default='',
                        help='directory of annotated images, named to match the JSON entries')
    parser.add_argument('--asset-id', default='TRB-001')
    parser.add_argument('--site', default='Site - Location TBC')
    parser.add_argument('--operator', default='DRONE EDUTRAIN LLC')
    parser.add_argument('--model', dest='model_name', default=None,
                        help='model name to print on the cover')
    parser.add_argument('--title', default='WIND TURBINE INSPECTION REPORT')
    args = parser.parse_args()

    generate_report(
        json_path=args.json_path,
        annotated_dir=args.annotated_dir,
        output_pdf=args.output_pdf,
        asset_id=args.asset_id,
        site=args.site,
        operator=args.operator,
        model_name=args.model_name,
        title=args.title,
    )


if __name__ == '__main__':
    main()
