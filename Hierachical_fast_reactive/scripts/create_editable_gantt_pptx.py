#!/usr/bin/env python3
"""Create an editable PowerPoint version of the async chunk Gantt schematic.

The project figure itself is rendered with Matplotlib, but this file emits a
minimal PowerPoint Open XML package where every block, label, cell, and arrow is
an editable PPT shape.  It intentionally avoids python-pptx so the artifact can
be regenerated in the current environment without extra dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


EMU_PER_INCH = 914_400
PT_TO_EMU = 12_700

SLIDE_W = 13.333
SLIDE_H = 7.5

BLACK = "222222"
GRAY = "666666"
LIGHT_GRID = "F4F4F4"
GRID_LINE = "E6E6E6"
BLUE = "0072B2"
BLUE_FILL = "DBEAFE"
BLUE_LIGHT = "E8F4FB"
GREEN = "009E73"
GREEN_FILL = "DFF1DF"
GREEN_LIGHT = "EAF7EE"
ORANGE = "E69F00"
ORANGE_FILL = "FFF4D6"
RED = "D55E00"
RED_FILL = "FFF1E6"
PURPLE = "CC79A7"


def emu(value: float) -> int:
    return int(round(value * EMU_PER_INCH))


def line_w(pt: float) -> int:
    return int(round(pt * PT_TO_EMU))


def color_xml(hex_color: str) -> str:
    return f'<a:solidFill><a:srgbClr val="{hex_color}"/></a:solidFill>'


def text_runs(
    text: str,
    *,
    size: float,
    color: str = BLACK,
    bold: bool = False,
    align: str = "ctr",
) -> str:
    bold_attr = ' b="1"' if bold else ""
    paragraphs: list[str] = []
    for raw_line in text.split("\n"):
        line = escape(raw_line)
        paragraphs.append(
            f'<a:p><a:pPr algn="{align}"/>'
            f'<a:r><a:rPr lang="en-US" sz="{int(size * 100)}"{bold_attr}>'
            f'{color_xml(color)}<a:latin typeface="Arial"/></a:rPr><a:t>{line}</a:t></a:r>'
            "</a:p>"
        )
    return "".join(paragraphs)


def tx_body(
    text: str,
    *,
    size: float = 10,
    color: str = BLACK,
    bold: bool = False,
    align: str = "ctr",
    margin: float = 0.03,
) -> str:
    m = emu(margin)
    return (
        f'<p:txBody><a:bodyPr wrap="square" anchor="mid" lIns="{m}" rIns="{m}" '
        f'tIns="{m}" bIns="{m}"/><a:lstStyle/>'
        f"{text_runs(text, size=size, color=color, bold=bold, align=align)}</p:txBody>"
    )


class SlideBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.shape_id = 2

    def next_id(self) -> int:
        sid = self.shape_id
        self.shape_id += 1
        return sid

    def textbox(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        size: float = 10,
        color: str = BLACK,
        bold: bool = False,
        align: str = "ctr",
        name: str = "Text",
    ) -> None:
        sid = self.next_id()
        self.parts.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/>'
            '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '<a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
            f"{tx_body(text, size=size, color=color, bold=bold, align=align)}</p:sp>"
        )

    def shape(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        fill: str | None = "FFFFFF",
        outline: str = BLACK,
        lw: float = 1.2,
        rounded: bool = False,
        size: float = 9,
        color: str = BLACK,
        bold: bool = False,
        align: str = "ctr",
        name: str = "Block",
        transparency: int | None = None,
    ) -> None:
        sid = self.next_id()
        geom = "roundRect" if rounded else "rect"
        if fill is None:
            fill_xml = "<a:noFill/>"
        else:
            alpha = f'<a:alpha val="{transparency}"/>' if transparency is not None else ""
            fill_xml = f'<a:solidFill><a:srgbClr val="{fill}">{alpha}</a:srgbClr></a:solidFill>'
        tx = tx_body(text, size=size, color=color, bold=bold, align=align) if text else "<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>"
        self.parts.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/>'
            '<p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>{fill_xml}'
            f'<a:ln w="{line_w(lw)}">{color_xml(outline)}</a:ln></p:spPr>{tx}</p:sp>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = BLACK,
        lw: float = 1.0,
        head: bool = False,
        tail: bool = False,
        dash: bool = False,
        name: str = "Line",
    ) -> None:
        sid = self.next_id()
        x = min(x1, x2)
        y = min(y1, y2)
        w = max(abs(x2 - x1), 0.001)
        h = max(abs(y2 - y1), 0.001)
        flip_h = ' flipH="1"' if x2 < x1 else ""
        flip_v = ' flipV="1"' if y2 < y1 else ""
        dash_xml = '<a:prstDash val="dash"/>' if dash else ""
        # In PowerPoint's line geometry, headEnd is the start point and tailEnd
        # is the end point.  The public arguments use normal arrow semantics.
        head_xml = '<a:tailEnd type="triangle"/>' if head else ""
        tail_xml = '<a:headEnd type="triangle"/>' if tail else ""
        self.parts.append(
            f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{sid}" name="{escape(name)}"/>'
            '<p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>'
            f'<p:spPr><a:xfrm{flip_h}{flip_v}><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            '<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
            f'<a:ln w="{line_w(lw)}">{color_xml(color)}{dash_xml}{head_xml}{tail_xml}</a:ln>'
            "</p:spPr><p:style/><p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:cxnSp>"
        )

    def double_arrow_label(
        self,
        x1: float,
        x2: float,
        y: float,
        label: str,
        *,
        color: str,
        size: float = 9,
        y_label_offset: float = -0.24,
    ) -> None:
        self.line(x1, y, x2, y, color=color, lw=1.4, head=True, tail=True)
        width = max(0.5, min(max(x2 - x1, 0.5), len(label) * 0.075))
        self.textbox((x1 + x2) / 2 - width / 2, y + y_label_offset, width, 0.18, label, size=size, color=color, bold=True)

    def xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            "<p:cSld><p:spTree>"
            '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            + "".join(self.parts)
            + "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
        )


def add_execution_cells(
    slide: SlideBuilder,
    x: float,
    y: float,
    width: float,
    *,
    steps: int,
    label: str,
    edge: str = GREEN,
    fill: str = GREEN_FILL,
) -> None:
    cell_w = width / steps
    for idx in range(steps):
        slide.shape(x + idx * cell_w, y, cell_w, 0.44, fill=fill, outline=edge, lw=0.75, name=f"{label} step {idx + 1}")
    slide.shape(x, y, width, 0.44, fill=None, outline=edge, lw=1.8, name=f"{label} outline")
    slide.textbox(x, y + 0.1, width, 0.22, label, size=10, bold=True)
    slide.textbox(x + 0.03, y + 0.28, 0.25, 0.12, "1", size=7.5, color=GRAY)
    slide.textbox(x + width - 0.28, y + 0.28, 0.25, 0.12, str(steps), size=7.5, color=GRAY)


def add_plan_blocks(
    slide: SlideBuilder,
    x_req: float,
    x_inf: float,
    x_ready: float,
    y: float,
    request: str,
    *,
    ready_text: str,
) -> None:
    slide.shape(x_req, y, 0.62, 0.42, request, fill=ORANGE_FILL, outline=ORANGE, lw=1.7, rounded=True, size=7.2, bold=True)
    slide.shape(x_inf, y, x_ready - x_inf - 0.5, 0.42, "SmolVLA\ninference", fill=BLUE_FILL, outline=BLUE, lw=1.7, rounded=True, size=7.2, bold=True)
    slide.shape(x_ready - 0.5, y, 0.66, 0.42, ready_text, fill=GREEN_LIGHT, outline=GREEN, lw=1.7, rounded=True, size=7.0, bold=True)


def build_problem_slide() -> str:
    slide = SlideBuilder()

    slide.textbox(0.35, 0.2, 6.8, 0.35, "Gantt view of async action chunks", size=19, bold=True, align="l")
    slide.textbox(
        0.35,
        0.58,
        11.9,
        0.28,
        "Planning starts during the tail of the active chunk; execution stays back-to-back, but each segment is open loop.",
        size=9.5,
        color=GRAY,
        align="l",
    )

    # Time grid.
    grid_x = 2.15
    grid_y = 1.0
    step = 0.35
    for i in range(30):
        fill = LIGHT_GRID if i % 2 == 0 else "FFFFFF"
        slide.shape(grid_x + i * step, grid_y, step, 5.85, fill=fill, outline=GRID_LINE, lw=0.25, name="time grid")
    slide.line(grid_x - 0.1, 6.95, grid_x + 30.4 * step, 6.95, color=BLACK, lw=1.0, head=True)
    slide.textbox(grid_x - 0.02, 7.02, 0.55, 0.18, "time", size=8.5, color=BLACK, align="l")

    # Row labels.
    rows = {
        "A": (1.34, 1.96),
        "B": (2.84, 3.46),
        "C": (4.34, 4.96),
    }
    for chunk, (plan_y, exec_y) in rows.items():
        slide.textbox(0.34, plan_y + 0.06, 0.9, 0.22, f"chunk {chunk}", size=12, bold=True, align="l")
        slide.textbox(1.55, plan_y + 0.06, 0.55, 0.18, "plan", size=8.5, color=BLUE, bold=True)
        slide.textbox(1.42, exec_y + 0.06, 0.7, 0.18, "execute", size=8.5, color=GREEN, bold=True)

    observe_a = 2.25
    infer_a = 3.02
    ready_a = 4.4
    exec_a_start = 4.65
    exec_w = 3.2
    exec_a_end = exec_a_start + exec_w
    request_b = exec_a_end - 1.95
    infer_b = request_b + 0.66
    ready_b = exec_a_end
    exec_b_end = ready_b + exec_w
    request_c = exec_b_end - 1.95
    infer_c = request_c + 0.66
    ready_c = exec_b_end

    # Timing variables.
    slide.double_arrow_label(infer_a, ready_a, 0.86, "d", color=RED, size=9, y_label_offset=0.02)
    slide.double_arrow_label(exec_a_start, exec_a_start + 2.8, 1.18, "H", color=BLUE, size=9, y_label_offset=-0.23)
    for i in range(10):
        slide.shape(exec_a_start + i * 0.28, 1.28, 0.28, 0.18, fill="EFF7FF", outline=BLUE, lw=0.6, name="planned horizon cell")
    slide.double_arrow_label(request_b, request_c, 0.92, "K", color=PURPLE, size=9, y_label_offset=-0.23)

    # Tail-overlap regions behind active execution bars.
    slide.shape(request_b, 1.9, ready_b - request_b, 0.64, fill=BLUE_LIGHT, outline=BLUE, lw=0.7, transparency=35000, name="tail overlap B planning")
    slide.line(request_b, 1.8, request_b, 2.63, color=BLUE, lw=0.8, dash=True)
    slide.line(ready_b, 1.8, ready_b, 2.63, color=BLUE, lw=0.8, dash=True)
    slide.textbox(request_b + 0.02, 1.74, ready_b - request_b - 0.04, 0.18, "tail-overlap planning", size=8.5, color=BLUE, bold=True)

    slide.shape(request_c, 3.4, ready_c - request_c, 0.64, fill=BLUE_LIGHT, outline=BLUE, lw=0.7, transparency=35000, name="tail overlap C planning")
    slide.line(request_c, 3.3, request_c, 4.13, color=BLUE, lw=0.8, dash=True)
    slide.line(ready_c, 3.3, ready_c, 4.13, color=BLUE, lw=0.8, dash=True)
    slide.textbox(request_c + 0.02, 3.24, ready_c - request_c - 0.04, 0.18, "tail-overlap planning", size=8.5, color=BLUE, bold=True)

    # Planning blocks.
    add_plan_blocks(slide, observe_a, infer_a, ready_a, rows["A"][0], "observe\nscene", ready_text="chunk A\nready")
    add_plan_blocks(slide, request_b, infer_b, ready_b, rows["B"][0], "request\nB", ready_text="chunk B\nready")
    add_plan_blocks(slide, request_c, infer_c, ready_c, rows["C"][0], "request\nC", ready_text="chunk C\nready")

    # Execution bars.
    add_execution_cells(slide, exec_a_start, rows["A"][1], exec_w, steps=8, label="execute A")
    add_execution_cells(slide, ready_b, rows["B"][1], exec_w, steps=8, label="execute B")
    add_execution_cells(slide, ready_c, rows["C"][1], 2.2, steps=6, label="future C", fill="F7FFF7")

    slide.double_arrow_label(exec_a_start, exec_a_end, rows["A"][1] + 0.68, "e=8 open-loop steps", color=GREEN, size=7.8, y_label_offset=-0.22)
    slide.double_arrow_label(ready_b, exec_b_end, rows["B"][1] + 0.68, "e=8 open-loop steps", color=GREEN, size=7.8, y_label_offset=-0.22)
    slide.textbox(ready_b + 0.04, rows["B"][1] - 0.24, 0.55, 0.18, "no gap", size=8, color=BLACK, align="l")
    slide.textbox(ready_c + 0.04, rows["C"][1] - 0.24, 0.55, 0.18, "no gap", size=8, color=BLACK, align="l")
    slide.textbox(ready_c + 0.1, rows["C"][1] + 0.56, 2.1, 0.18, "prepared next segment", size=8.5, color=GREEN, bold=True)

    # Feedback row.
    slide.textbox(0.35, 5.88, 0.95, 0.35, "feedback\nstate", size=12, color=RED, bold=True, align="l")
    fb_y = 6.05
    slide.shape(4.15, fb_y, 1.45, 0.48, "visual feedback\nat chunk start", fill=RED_FILL, outline=RED, lw=1.5, rounded=True, size=7.6, bold=True)
    slide.shape(5.95, fb_y, 2.4, 0.48, "middle steps use\nstale feedback", fill=RED_FILL, outline=RED, lw=1.5, rounded=True, size=8.0, bold=True)
    slide.shape(8.75, fb_y, 1.55, 0.48, "local mismatch\ncan grow", fill=RED_FILL, outline=RED, lw=1.5, rounded=True, size=7.6, bold=True)
    slide.line(5.6, fb_y + 0.24, 5.92, fb_y + 0.24, color=RED, lw=1.1, head=True)
    slide.line(8.35, fb_y + 0.24, 8.72, fb_y + 0.24, color=RED, lw=1.1, head=True)
    for cx in [exec_a_start + 0.55, exec_a_start + 1.15, exec_a_start + 1.75, exec_a_start + 2.35, ready_b + 0.55, ready_b + 1.15, ready_b + 1.75, ready_b + 2.35]:
        slide.line(cx, 5.55, cx, 5.87, color="B8B8B8", lw=0.7)

    # Legend.
    slide.shape(10.25, 1.06, 2.55, 1.06, fill="FFFFFF", outline="D9D9D9", lw=0.8, name="legend")
    legend_items = [
        ("H:", "planned action horizon"),
        ("e:", "open-loop steps executed"),
        ("K:", "time between replans"),
        ("d:", "SmolVLA compute delay"),
    ]
    for idx, (sym, desc) in enumerate(legend_items):
        y = 1.2 + idx * 0.23
        slide.textbox(10.38, y, 0.28, 0.15, sym, size=8.2, bold=True, align="l")
        slide.textbox(10.67, y, 1.85, 0.15, desc, size=8.2, color=GRAY, align="l")

    # Small editing note kept outside the figure content.
    slide.textbox(10.25, 6.95, 2.85, 0.2, "All elements are editable PowerPoint shapes.", size=7.5, color=GRAY, align="r")

    return slide.xml()


def add_small_cells(slide: SlideBuilder, x: float, y: float, n: int, *, w: float, h: float, edge: str, fill: str) -> None:
    for i in range(n):
        slide.shape(x + i * w, y, w, h, fill=fill, outline=edge, lw=0.7, name="action cell")


def build_architecture_slide() -> str:
    slide = SlideBuilder()

    slide.textbox(0.35, 0.18, 7.9, 0.35, "HFRVLA architecture: slow chunk, fast wrist residual", size=18, bold=True, align="l")
    slide.textbox(
        0.35,
        0.55,
        11.9,
        0.28,
        "Frozen SmolVLA proposes a base action chunk; a trainable wrist residual corrects the selected action before robot control.",
        size=9.2,
        color=GRAY,
        align="l",
    )

    # Slow planner branch.
    slide.textbox(0.45, 1.02, 2.2, 0.18, "low-frequency slow planner", size=9.2, color=BLUE, bold=True, align="l")
    slide.shape(0.45, 1.32, 1.2, 0.5, "language\ninstruction", fill="EEF5FB", outline=BLUE, lw=1.4, rounded=True, size=8.2, bold=True)
    slide.shape(0.45, 2.06, 1.35, 0.72, "top RGB\nwrist RGB\n+ state", fill="F5F5F5", outline="777777", lw=1.2, rounded=True, size=8.0, bold=True)
    slide.shape(2.35, 1.68, 2.05, 0.94, "Frozen SmolVLA\nVLM + action expert", fill=BLUE_FILL, outline=BLUE, lw=1.7, rounded=True, size=10, bold=True)
    slide.textbox(2.42, 2.65, 1.6, 0.16, "frozen slow planner", size=7.2, color=BLUE, bold=True, align="l")
    slide.shape(4.95, 1.35, 1.5, 0.48, "base action\nchunk", fill="FFFFFF", outline=BLUE, lw=1.5, rounded=True, size=8.4, bold=True)
    slide.shape(4.95, 2.16, 1.5, 0.5, "slow context\n+ chunk index", fill=ORANGE_FILL, outline=ORANGE, lw=1.3, rounded=True, size=8.0, bold=True)
    slide.line(1.66, 1.57, 2.35, 1.92, color=BLUE, lw=1.1, head=True)
    slide.line(1.81, 2.42, 2.35, 2.22, color="777777", lw=1.0, head=True)
    slide.line(4.4, 1.93, 4.95, 1.59, color=BLUE, lw=1.1, head=True)
    slide.line(4.4, 2.22, 4.95, 2.38, color=ORANGE, lw=1.0, head=True)

    # Queued base actions and selected current action.
    add_small_cells(slide, 7.0, 1.42, 8, w=0.25, h=0.22, edge=BLUE, fill="F2F7FD")
    slide.shape(7.0, 1.42, 0.25, 0.22, fill=GREEN_FILL, outline=BLUE, lw=0.7)
    slide.textbox(7.03, 1.13, 2.0, 0.18, "queued base actions", size=8.0, color=BLUE, bold=True)
    slide.shape(7.35, 2.08, 1.45, 0.48, "selected\nbase action", fill="FFFFFF", outline=BLUE, lw=1.4, rounded=True, size=8.2, bold=True)
    slide.line(6.45, 1.58, 7.0, 1.53, color=BLUE, lw=1.0, head=True)
    slide.line(7.18, 1.64, 7.85, 2.08, color=BLUE, lw=1.0, head=True)

    # Fast wrist residual branch.
    slide.textbox(0.45, 3.27, 2.35, 0.18, "high-frequency wrist correction", size=9.2, color=GREEN, bold=True, align="l")
    slide.shape(0.45, 3.58, 1.22, 0.52, "current\nwrist view", fill=GREEN_LIGHT, outline=GREEN, lw=1.5, rounded=True, size=8.6, bold=True)
    slide.shape(2.18, 3.5, 1.72, 0.66, "Frozen DINOv3\nwrist patches", fill=GREEN_LIGHT, outline=GREEN, lw=1.5, rounded=True, size=8.8, bold=True)
    slide.textbox(2.25, 4.18, 1.4, 0.16, "frozen feature extractor", size=7.0, color=GREEN, bold=True, align="l")
    slide.shape(4.9, 3.34, 2.05, 0.94, "Trainable fast\nwrist residual\nmodule", fill="DFF3E9", outline=GREEN, lw=1.8, rounded=True, size=10, bold=True)
    slide.shape(4.75, 4.6, 0.8, 0.36, "state", fill="F5F5F5", outline="777777", lw=1.0, rounded=True, size=8.2, bold=True)
    slide.line(1.67, 3.84, 2.18, 3.83, color=GREEN, lw=1.1, head=True)
    slide.line(3.9, 3.83, 4.9, 3.78, color=GREEN, lw=1.1, head=True)
    slide.line(5.15, 4.6, 5.15, 4.28, color="777777", lw=0.9, head=True)
    slide.line(6.45, 2.42, 5.55, 3.34, color=ORANGE, lw=1.0, head=True)
    slide.line(8.07, 2.56, 6.5, 3.34, color=BLUE, lw=1.0, head=True)

    # Bounded residual merge.
    slide.shape(7.55, 3.58, 1.1, 0.52, "residual\ncorrection", fill="FFFFFF", outline=GREEN, lw=1.4, rounded=True, size=7.8, bold=True)
    slide.shape(9.05, 3.53, 1.15, 0.62, "clip +\nscale", fill="FFFFFF", outline=GREEN, lw=1.4, rounded=True, size=8.4, bold=True)
    slide.shape(10.75, 3.58, 0.46, 0.46, "+", fill="FFFFFF", outline=BLACK, lw=1.2, rounded=True, size=16, bold=True)
    slide.shape(11.75, 3.5, 1.15, 0.64, "final\naction", fill="F7F7F7", outline=BLACK, lw=1.3, rounded=True, size=9.0, bold=True)
    slide.shape(11.75, 5.0, 1.15, 0.64, "control\nrobot", fill="F5F5F5", outline=BLACK, lw=1.3, rounded=True, size=9.0, bold=True)
    slide.line(6.95, 3.8, 7.55, 3.84, color=GREEN, lw=1.2, head=True)
    slide.line(8.65, 3.84, 9.05, 3.84, color=GREEN, lw=1.1, head=True)
    slide.line(10.2, 3.84, 10.75, 3.84, color=GREEN, lw=1.1, head=True)
    slide.line(8.8, 2.34, 10.75, 3.73, color=BLUE, lw=1.0, head=True)
    slide.line(11.21, 3.82, 11.75, 3.82, color=BLACK, lw=1.1, head=True)
    slide.line(12.32, 4.14, 12.32, 5.0, color=BLACK, lw=1.1, head=True)

    # Feedback line from robot to wrist view, drawn as editable segmented lines.
    slide.line(11.75, 5.32, 1.05, 5.32, color=GREEN, lw=0.9)
    slide.line(1.05, 5.32, 1.05, 4.1, color=GREEN, lw=0.9, head=True)
    slide.textbox(3.9, 5.38, 3.9, 0.18, "current wrist feedback supplies per-step local evidence", size=8.0, color=GREEN, bold=True)

    # Short formula and legend.
    slide.shape(7.05, 5.68, 3.55, 0.52, "a_final = a_base + alpha * clip(delta_a)", fill="FFFFFF", outline="D9D9D9", lw=0.9, rounded=True, size=10.2, bold=True)
    slide.shape(0.45, 6.35, 2.0, 0.42, "blue/orange: frozen slow context", fill="FFFFFF", outline=BLUE, lw=1.0, rounded=True, size=7.7, color=BLUE, bold=True)
    slide.shape(2.72, 6.35, 2.0, 0.42, "green: trainable fast residual", fill="FFFFFF", outline=GREEN, lw=1.0, rounded=True, size=7.7, color=GREEN, bold=True)
    slide.textbox(10.05, 6.95, 3.0, 0.2, "All elements are editable PowerPoint shapes.", size=7.5, color=GRAY, align="r")

    return slide.xml()


def rels_root() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def content_types() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        '<Override PartName="/ppt/slides/slide2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
        "</Types>"
    )


def presentation_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        '<p:sldIdLst><p:sldId id="256" r:id="rId2"/><p:sldId id="257" r:id="rId3"/></p:sldIdLst>'
        f'<p:sldSz cx="{emu(SLIDE_W)}" cy="{emu(SLIDE_H)}" type="screen16x9"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        '<p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle>'
        "</p:presentation>"
    )


def presentation_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/>'
        "</Relationships>"
    )


def slide_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        "</Relationships>"
    )


def master_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
        '<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>'
        '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/>'
        '<a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>'
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
        'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>'
        "</p:sldMaster>"
    )


def master_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
        "</Relationships>"
    )


def layout_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">'
        '<p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
        '<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>'
        '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/>'
        '<a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>'
        '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    )


def layout_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
        "</Relationships>"
    )


def theme_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="HFRVLA">'
        '<a:themeElements><a:clrScheme name="HFRVLA">'
        '<a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>'
        '<a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>'
        f'<a:accent1><a:srgbClr val="{BLUE}"/></a:accent1><a:accent2><a:srgbClr val="{GREEN}"/></a:accent2>'
        f'<a:accent3><a:srgbClr val="{ORANGE}"/></a:accent3><a:accent4><a:srgbClr val="{RED}"/></a:accent4>'
        f'<a:accent5><a:srgbClr val="{PURPLE}"/></a:accent5><a:accent6><a:srgbClr val="666666"/></a:accent6>'
        '<a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink>'
        '</a:clrScheme><a:fontScheme name="Arial"><a:majorFont><a:latin typeface="Arial"/></a:majorFont>'
        '<a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme><a:fmtScheme name="Default">'
        '<a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
        '<a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
        '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
        '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>'
        '</a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>'
    )


def core_xml() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:title>Editable HFRVLA Async Chunk Gantt</dc:title><dc:creator>Codex</dc:creator>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def app_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>Codex</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat>'
        '<Slides>2</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides></Properties>'
    )


def write_pptx(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = {
        "[Content_Types].xml": content_types(),
        "_rels/.rels": rels_root(),
        "docProps/core.xml": core_xml(),
        "docProps/app.xml": app_xml(),
        "ppt/presentation.xml": presentation_xml(),
        "ppt/_rels/presentation.xml.rels": presentation_rels(),
        "ppt/slides/slide1.xml": build_problem_slide(),
        "ppt/slides/_rels/slide1.xml.rels": slide_rels(),
        "ppt/slides/slide2.xml": build_architecture_slide(),
        "ppt/slides/_rels/slide2.xml.rels": slide_rels(),
        "ppt/slideMasters/slideMaster1.xml": master_xml(),
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": master_rels(),
        "ppt/slideLayouts/slideLayout1.xml": layout_xml(),
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": layout_rels(),
        "ppt/theme/theme1.xml": theme_xml(),
    }
    with ZipFile(out_path, "w", ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def main() -> None:
    out_path = Path("paper/src/figures/fig1_problem_schematic_editable.pptx")
    write_pptx(out_path)
    print(f"[editable-pptx] wrote {out_path}")


if __name__ == "__main__":
    main()
