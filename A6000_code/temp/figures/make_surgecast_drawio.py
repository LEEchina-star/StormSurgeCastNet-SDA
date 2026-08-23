# -*- coding: utf-8 -*-
"""Build a COMPLETE SDA-Diff framework drawio (training + inference + SDA + 2x SR + ensemble).
North-up, land-masked Gulf thumbnails, connected edges."""
import html, base64

PAL = dict(DATA=("#DCE9F7","#3E6B9E"), CONV=("#E3F0E6","#2F7D46"), NOISE=("#F5E6D3","#B06A1B"),
           DA=("#FBE3DE","#C0392B"), LOSS=("#EDE3F6","#6A3D9A"), UP=("#C8EAD3","#1E7A7A"),
           WHITE=("#FFFFFF","#BBBBBB"))

def b64(name):
    return "data:image/png;base64," + base64.b64encode(open(f"temp/figures/{name}","rb").read()).decode()

cells=[]; nid=2
def V(x,y,w,h,text,kind,bold=False,fs=12,fill=True,stroke_w=1.5,dashed=False):
    global nid
    f,s=PAL[kind]
    st=(f"rounded=1;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};strokeWidth={stroke_w};"
        f"fontSize={fs};fontColor=#222222;fontStyle={'1' if bold else '0'};shadow=1;"
        f"gradientColor=#FFFFFF;gradientDirection=north;{'fillColor=none;' if not fill else ''}"
        f"{'dashed=1;dashPattern=6 4;' if dashed else ''}")
    cells.append(f'<mxCell id="{nid}" value="{html.escape(text)}" style="{st}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    nid+=1; return nid-1
def IMG(x,y,w,h,name):
    global nid
    cells.append(f'<mxCell id="{nid}" value="" style="shape=image;aspect=fixed;shadow=1;html=1;image={b64(name)};" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    nid+=1; return nid-1
def E(src,tgt,color="#444444",dashed=False,width=1.4):
    global nid
    st=f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeColor={color};strokeWidth={width};{('dashed=1;' if dashed else '')}"
    cells.append(f'<mxCell id="{nid}" value="" style="{st}" edge="1" parent="1" source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    nid+=1; return nid-1

# ================= INPUT =================
V(30,15,470,40,"<b>Input</b>","DATA",bold=True,fs=15)
m_geo=IMG(40,60,225,240,"gulf3_map.png")
t_sp=IMG(40,315,145,145,"gulf_thumb_sparse.png")
t_ms=IMG(200,315,145,145,"gulf_thumb_msl.png")
t_wd=IMG(40,470,145,145,"gulf_thumb_wind.png")
t_gt=IMG(200,470,145,145,"gulf_thumb_gtsm.png")
cbox=V(40,625,420,52,"<b>c = [B, T=12, 6, H, W]</b> + lead L<br><font size=\"2\">(land-masked, north-up)</font>","DATA",bold=True,fs=12)
for t in [t_sp,t_ms,t_wd,t_gt]: E(t,cbox,color="#3E6B9E",width=1.1)

# ================= MODEL =================
V(560,15,1420,40,"<b>Model — conditional diffusion (EDM) + SDA data assimilation</b>","CONV",bold=True,fs=15)
ctx=V(560,70,420,80,"<b>Context encoder</b><br>FiLM / attention-group,<br>time &amp; lead embeddings","CONV",fs=11)
E(cbox,ctx,color="#2F7D46")

# -------- training (upper, dashed) --------
V(560,180,1420,330,"","WHITE",dashed=True)
V(575,188,1390,30,"<b>Training — learn to recover the surge field from noise</b>","CONV",bold=True,fs=12,dashed=True)
x0=V(590,240,150,80,"<b>x\u2080</b> (clean surge, 256\u00b2)<br><font size=\"2\">dense target</font>","DATA",fs=11)
nz=V(790,240,180,80,"<b>x\u209c = x\u2080 + \u03c3\u03b5</b><br><font size=\"2\">forward noising, \u03c3 ~ p(\u03c3)</font>","NOISE",bold=True,fs=11)
dn=V(1020,230,260,100,"<b>D\u03b8(x\u209c, \u03c3, c, L)</b><br>conditional denoiser (UNet)<br><font size=\"2\">EDM preconditioning, FiLM</font>","CONV",bold=True,fs=11)
xh=V(1330,240,150,80,"<b>x\u0302\u2080</b> (estimate)","CONV",fs=11)
proc=IMG(1520,210,300,130,"gulf3_denoise.png")
loss=V(590,430,1230,60,"<b>L = E[\u03bb(\u03c3) \u2016 D\u03b8(x\u2080+\u03c3\u03b5, \u03c3, c, L) \u2212 x\u2080 \u2016\u00b2]</b>  <font size=\"2\">(weighted, valid pixels only)</font>","LOSS",bold=True,fs=12)
E(x0,nz,color="#B06A1B"); E(nz,dn,color="#2F7D46"); E(dn,xh,color="#2F7D46")
E(ctx,dn,color="#2F7D46",dashed=True)
E(xh,proc,color="#2F7D46",width=1.1); E(proc,loss,color="#6A3D9A",dashed=True)

# -------- inference (lower, dashed) --------
V(560,540,1420,380,"","WHITE",dashed=True)
V(575,548,1390,30,"<b>Inference — reverse diffusion + SDA likelihood guidance (assimilate new observations)</b>","DA",bold=True,fs=12,dashed=True)
nz2=V(590,600,140,70,"<b>z ~ N(0,I)</b><br>noise seed","NOISE",fs=11)
prior=V(780,600,170,70,"<b>prior p(x|c)</b><br>reverse diffusion","CONV",fs=11)
sda=V(1000,590,300,110,"<b>\u2207log p(x\u209c|y,c)</b><br>= s\u03b8(x\u209c,\u03c3,c) + \u2207log N(y|A x\u0302\u2080, R)<br><font size=\"2\">annealed 1/(R\u00b2+\u03c3\u00b2)</font>","DA",bold=True,fs=10)
yimg=IMG(620,700,130,130,"gulf_thumb_sparse.png")
ybox=V(760,715,220,60,"<b>y</b>: GESLA-3 obs<br><b>A</b>: valid_mask, <b>R</b>: obs noise","DATA",fs=9)
post=V(1350,600,180,70,"<b>posterior p(x|c,y)</b>","DA",bold=True,fs=11)
proc2=IMG(1560,580,330,135,"gulf_assim.png")
E(nz2,prior,color="#2F7D46"); E(prior,sda,color="#C0392B"); E(sda,post,color="#C0392B")
E(yimg,ybox,color="#3E6B9E",width=1.0); E(ybox,sda,color="#C0392B")
E(ctx,sda,color="#2F7D46",dashed=True)
E(post,proc2,color="#C0392B",width=1.1)

# 2x super-resolution marker
sr=V(560,945,1420,50,"<b>Generative super-resolution (2\u00d7)</b> — coarse GTSM / gauge obs \u2192 dense 512\u00d7512 field (land-masked)",
    "UP",bold=True,fs=12)
E(post,sr,color="#1E7A7A",dashed=True)

# ================= OUTPUT =================
V(2010,15,430,40,"<b>Output</b>","DATA",bold=True,fs=15)
ens=V(2010,70,430,50,"<b>N ensemble samples</b><br><font size=\"2\">seed-dependent reverse diffusion</font>","UP",fs=10)
o1=IMG(2010,150,200,200,"gulf_thumb_mean.png")
o2=IMG(2240,150,200,200,"gulf_thumb_interval.png")
o3=IMG(2010,380,200,200,"gulf_thumb_exceed.png")
V(2010,355,200,22,"mean","UP",fs=9)
V(2240,355,200,22,"90% interval","UP",fs=9)
V(2010,585,200,22,"P(surge&gt;0.5 m)","UP",fs=9)
E(post,ens,color="#1E7A7A"); E(ens,o1,color="#1E7A7A"); E(ens,o2,color="#1E7A7A"); E(ens,o3,color="#1E7A7A")

xml=f'''<mxfile host="app.diagrams.net" agent="SDA-Diff" version="24.0.0"><diagram id="sdadiff" name="SDA-Diff framework"><mxGraphModel dx="2000" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2460" pageHeight="1030" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{''.join(cells)}</root></mxGraphModel></diagram></mxfile>'''
out="/Volumes/2022-docs/科研工作2025/CCMP-2024/2025/SurgeCast(1)_v2.drawio"
open(out,"w").write(xml)
print(f"generated {out}: {len(cells)} cells, {nid} ids")
