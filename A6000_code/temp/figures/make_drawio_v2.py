# -*- coding: utf-8 -*-
"""draw.io v2: 3D-shadow rounded boxes, real-data thumbnails, concise labels."""
import html, base64, os

PAL = dict(
    DATA=("#DCE9F7", "#3E6B9E"), CONV=("#E3F0E6", "#2F7D46"), DOWN=("#CDE3F5", "#1A5A9E"),
    UP=("#C8EAD3", "#1E7A7A"), NOISE=("#F5E6D3", "#B06A1B"), DA=("#FBE3DE", "#C0392B"),
    LOSS=("#EDE3F6", "#6A3D9A"), WHITE=("#FFFFFF", "#BBBBBB"))

def b64(name):
    with open(f"temp/figures/{name}", "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

cells = []; nid = 2

def V(x, y, w, h, text, kind, bold=False, fs=13, fill=True, tc=None, stroke_w=1.5, shadow=True):
    global nid
    f, s = PAL[kind]
    style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};strokeWidth={stroke_w};"
             f"fontSize={fs};fontColor={'#222222' if tc is None else tc};fontStyle={'1' if bold else '0'};"
             f"{'shadow=1;' if shadow else ''}gradientColor=#FFFFFF;gradientDirection=north;"
             f"{'fillColor=none;' if not fill else ''}")
    cells.append(f'<mxCell id="{nid}" value="{html.escape(text)}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    nid += 1; return nid-1

def IMG(x, y, w, h, name):
    global nid
    style = f"shape=image;aspect=fixed;shadow=1;html=1;image={b64(name)};"
    cells.append(f'<mxCell id="{nid}" value="" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    nid += 1; return nid-1

def E(src, tgt, color="#444444", dashed=0, width=1.4):
    global nid
    style = f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeColor={color};strokeWidth={width};{('dashed=1;' if dashed else '')}"
    cells.append(f'<mxCell id="{nid}" value="" style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    nid += 1; return nid-1

# ===== INPUT (left, thumbnails) =====
V(40, 20, 360, 34, "Input", "DATA", bold=True, fs=15)
i1 = IMG(50, 65, 150, 150, "thumb_sparse.png")
i2 = IMG(210, 65, 150, 150, "thumb_era5_msl.png")
i3 = IMG(50, 225, 150, 150, "thumb_wind.png")
i4 = IMG(210, 225, 150, 150, "thumb_gtsm.png")
V(40, 390, 360, 30, "sparse gauges", "DATA", fs=11, shadow=False)
V(40, 390+0, 360, 0, "", "WHITE", shadow=False)  # spacer ignored
c_box = V(40, 440, 360, 44, "<b>c = [B,T,6,H,W]</b> + lead L", "DATA", bold=True, fs=12)
for t in [i1,i2,i3,i4]:
    E(t, c_box, color="#3E6B9E", width=1.2)

# ===== MODEL (centre) =====
V(440, 20, 720, 470, "", "WHITE", fill=False, stroke_w=1.0, shadow=False)
V(440, 20, 720, 34, "Model — EDM generation + SDA assimilation", "CONV", bold=True, fs=15)

# training (upper)
V(460, 70, 680, 165, "", "WHITE", fill=False, shadow=False)
V(475, 78, 650, 30, "Training — learn to recover the surge field from noise", "CONV", bold=True, fs=12, fill=False, tc="#2F7D46")
x0 = V(470, 118, 100, 64, "x<sub>0</sub>", "DATA", fs=13)
nz = V(600, 118, 140, 64, "x<sub>t</sub> = x<sub>0</sub>+σε", "NOISE", fs=12)
unet = V(770, 112, 180, 76, "<b>D<sub>θ</sub></b> U-Net<br>conv·down·up", "CONV", fs=12)
xhat = V(980, 118, 100, 64, "x̂<sub>0</sub>", "CONV", fs=13)
E(x0, nz, color="#B06A1B"); E(nz, unet); E(unet, xhat)
V(600, 195, 200, 26, "FiLM (σ, L, c)", "NOISE", fs=10, shadow=False)
E(unet, xhat)
loss = V(470, 250, 610, 40, "<b>L = E[ λ(σ)‖D<sub>θ</sub>(x<sub>0</sub>+σε)−x<sub>0</sub>‖² ]</b>", "LOSS", bold=True, fs=12)
E(unet, loss, color="#6A3D9A", dashed=1)

# inference (lower, red)
V(460, 320, 680, 160, "", "WHITE", fill=False, shadow=False)
V(475, 328, 650, 30, "Inference — correct the sampling path with new observations", "DA", bold=True, fs=12, fill=False, tc="#C0392B")
nx = V(470, 370, 90, 56, "noise", "DATA", fs=12)
den = V(590, 370, 120, 56, "denoise", "CONV", fs=12)
tw = V(740, 370, 120, 56, "Tweedie x̂<sub>0</sub>", "CONV", fs=12)
sda = V(560, 445, 460, 40, "<b>∇log p(x<sub>t</sub>|y,c) = s<sub>θ</sub> + ∇log N(y|Ax̂<sub>0</sub>,R)</b>", "DA", bold=True, fs=12, tc="#C0392B", stroke_w=2.5)
obs = V(900, 370, 170, 56, "obs. y (gauge)", "DATA", fs=12)
E(nx, den); E(den, tw); E(tw, sda, color="#C0392B"); E(obs, sda, color="#C0392B", dashed=1)

# ===== OUTPUT (right, thumbnails) =====
V(1200, 20, 360, 470, "", "WHITE", fill=False, shadow=False)
V(1200, 20, 360, 34, "Output (ensemble)", "DATA", bold=True, fs=15)
o1 = IMG(1210, 70, 160, 160, "thumb_out_mean.png")
o2 = IMG(1380, 70, 160, 160, "thumb_out_interval.png")
o3 = IMG(1210, 245, 160, 160, "thumb_out_exceed.png")
V(1210, 415, 160, 26, "mean", "UP", fs=11, shadow=False)
V(1380, 415, 160, 26, "90% interval", "UP", fs=11, shadow=False)
V(1210, 448, 330, 26, "P(surge &gt; h)", "UP", fs=11, shadow=False)
E(c_box, unet, color="#2F7D46")
E(xhat, o1, color="#1E7A7A")
E(sda, o1, color="#C0392B")

xml = f'''<mxfile host="app.diagrams.net" agent="SDA-Diff" version="24.0.0">
  <diagram id="sdadiff" name="SDA-Diff framework v2">
    <mxGraphModel dx="1700" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1640" pageHeight="540" math="0" shadow="0">
      <root><mxCell id="0"/><mxCell id="1" parent="0"/>{''.join(cells)}</root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
open("temp/figures/fig1_sdadiff_v2.drawio","w").write(xml)
print(f"v2 生成：{len(cells)} 元素")
