# -*- coding: utf-8 -*-
"""Generate an editable draw.io (.drawio) framework figure for SDA-Diff,
mirroring the white-background professional style of reconswathnet.
Colour code: conv=green, downsample=blue, upsample=cyan, noise/drop=orange,
SDA assimilation=red (sole highlight), loss=purple, data=blue-grey."""
import html

# palette: (fill, stroke) light fill + saturated border
PAL = dict(
    DATA=("#DCE9F7", "#3E6B9E"),
    CONV=("#E3F0E6", "#2F7D46"),
    DOWN=("#CDE3F5", "#1A5A9E"),
    UP=("#C8EAD3", "#1E7A7A"),
    NOISE=("#F5E6D3", "#B06A1B"),
    DA=("#FBE3DE", "#C0392B"),
    LOSS=("#EDE3F6", "#6A3D9A"),
    WHITE=("#FFFFFF", "#AAAAAA"),
)

cells = []
edges = []
nid = 2

def V(x, y, w, h, text, kind, bold=False, fs=12, fill=True, rounded=1, dashed=0, tc=None, stroke_w=1.5):
    global nid
    f, s = PAL[kind]
    style = (f"rounded={rounded};whiteSpace=wrap;html=1;"
             f"fillColor={f};strokeColor={s};strokeWidth={stroke_w};"
             f"fontSize={fs};fontColor={'#222222' if tc is None else tc};"
             f"fontStyle={'1' if bold else '0'};"
             f"{'dashed=1;' if dashed else ''}"
             f"{'fillColor=none;' if not fill else ''}")
    cells.append(f'<mxCell id="{nid}" value="{html.escape(text)}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    nid += 1
    return nid - 1

def E(src, tgt, color="#444444", dashed=0, width=1.3, label=""):
    global nid
    style = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;"
             f"strokeColor={color};strokeWidth={width};{('dashed=1;' if dashed else '')}")
    cells.append(f'<mxCell id="{nid}" value="{html.escape(label)}" style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    nid += 1
    return nid - 1

# ================= INPUT (left) =================
V(40, 30, 280, 40, "Input (same as StormSurgeCastNet / Ebel)", "DATA", bold=True, fs=14)
V(40, 80, 280, 70, "sparse in-situ gauges<br>+ valid mask  <i>(1…T)</i>", "DATA", fs=12)
V(40, 160, 280, 70, "ERA5  msl · u<sub>10</sub> · v<sub>10</sub>  <i>(1…T)</i>", "DATA", fs=12)
V(40, 240, 280, 70, "GTSM surge  <i>(1…T)</i>", "DATA", fs=12)
c_box = V(40, 330, 280, 70, "<b>c = [B, T, 6, H, W]</b><br>+ lead time L", "DATA", bold=False, fs=12)
for t in [2,3,4]:
    E(t, c_box, color="#3E6B9E")

# ================= MODEL (centre) =================
V(360, 30, 820, 470, "", "WHITE", fill=False, stroke_w=1.0)
V(360, 30, 820, 40, "Model — EDM generation + SDA assimilation", "CONV", bold=True, fs=14)

# --- training path (upper) ---
V(380, 80, 780, 175, "", "WHITE", fill=False, dashed=1)
V(395, 90, 750, 34, "Training — learn to recover the surge field from noise", "CONV", bold=True, fs=13, fill=False, tc="#2F7D46")
x0 = V(400, 135, 120, 70, "x<sub>0</sub><br>target", "DATA", fs=12)
noise = V(555, 135, 160, 70, "x<sub>t</sub> = x<sub>0</sub> + σε<br><i>(noise / drop)</i>", "NOISE", fs=12)
unet = V(750, 125, 200, 90, "<b>D<sub>θ</sub>(x<sub>t</sub>, σ, c, L)</b><br>U-Net<br>conv · down · up · attn", "CONV", bold=False, fs=12)
xhat = V(990, 135, 130, 70, "x̂<sub>0</sub> = D<sub>θ</sub>", "CONV", fs=12)
E(x0, noise, color="#B06A1B")
E(noise, unet)
E(unet, xhat)
# FiLM conditioning
film = V(555, 215, 200, 34, "FiLM: ν(σ) + lead L  +  context c", "NOISE", fs=10)
E(film, unet, color="#B06A1B", dashed=1)
# loss
loss = V(400, 260, 720, 55, "<b>L = E[ λ(σ) ‖ D<sub>θ</sub>(x<sub>0</sub>+σε, σ, c, L) − x<sub>0</sub> ‖² ]</b>  (masked, weighted)", "LOSS", bold=False, fs=12)
E(unet, loss, color="#6A3D9A", dashed=1)

# --- inference path (lower) ---
V(380, 335, 780, 160, "", "WHITE", fill=False, dashed=1)
V(395, 345, 750, 34, "Inference — correct the sampling path with new observations", "DA", bold=True, fs=13, fill=False, tc="#C0392B")
nx = V(400, 400, 110, 60, "noise<br>x<sub>T</sub>", "DATA", fs=12)
den = V(545, 400, 140, 60, "denoise<br>(EDM Heun)", "CONV", fs=12)
tw = V(720, 400, 130, 60, "Tweedie<br>x̂<sub>0</sub>", "CONV", fs=12)
obs = V(1000, 400, 160, 60, "obs. y, A, R<br>(gauge pixels)", "DATA", fs=11)
sda = V(600, 475, 420, 60, "<b>SDA:  ∇log p(x<sub>t</sub>|y,c) = s<sub>θ</sub> + ∇log N(y|Ax̂<sub>0</sub>, R)</b>", "DA", bold=True, fs=12, tc="#C0392B", stroke_w=2)
E(nx, den)
E(den, tw)
E(tw, sda, color="#C0392B")
E(obs, sda, color="#C0392B", dashed=1)

# ================= OUTPUT (right) =================
V(1220, 30, 300, 470, "", "WHITE", fill=False)
V(1220, 30, 300, 40, "Output (posterior ensemble)", "DATA", bold=True, fs=14)
ens = V(1235, 90, 270, 70, "posterior samples<br>{x<sub>0</sub><sup>(i)</sup>} (×N)", "CONV", fs=12)
stats = V(1235, 180, 270, 80, "<b>mean · quantiles</b><br>P(surge &gt; h)", "UP", fs=12)
dense = V(1235, 280, 270, 80, "<b>dense surge forecast</b><br>(uncertainty-aware)", "UP", bold=True, fs=12)
E(c_box, unet, color="#2F7D46")
E(xhat, stats, color="#1E7A7A")
E(sda, ens, color="#C0392B")
E(ens, stats)
E(stats, dense)

# ================= legend =================
lx, ly = 40, 440
leg = [("data / tensors","DATA"),("convolution / denoise","CONV"),("downsample","DOWN"),
       ("upsample / skip","UP"),("noise / drop / FiLM","NOISE"),("SDA assimilation (highlight)","DA"),("loss","LOSS")]
for i,(lab,k) in enumerate(leg):
    f,s = PAL[k]
    cells.append(f'<mxCell id="{nid}" value="{lab}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};fontSize=10;" vertex="1" parent="1">'
                 f'<mxGeometry x="{lx + i*180}" y="{ly}" width="170" height="26" as="geometry"/></mxCell>')
    nid += 1

xml = f'''<mxfile host="app.diagrams.net" agent="SDA-Diff" version="24.0.0">
  <diagram id="sdadiff" name="SDA-Diff framework">
    <mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="520" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {''.join(cells)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

open("temp/figures/fig1_sdadiff.drawio", "w").write(xml)
print(f"生成 fig1_sdadiff.drawio：{len(cells)} 个元素（模块+箭头+图例）")
