# -*- coding: utf-8 -*-
"""Two mechanism schematics for the SDA-Diff framework:
  (1) Context encoder: FiLM / attention-group + time & lead embeddings
  (2) Generative 2x super-resolution: coarse GTSM -> dense 512x512
Output: drawio + png, palette consistent with the framework."""
import html, base64, subprocess, os

PAL = dict(DATA=("#DCE9F7","#3E6B9E"), CONV=("#E3F0E6","#2F7D46"), NOISE=("#F5E6D3","#B06A1B"),
           DA=("#FBE3DE","#C0392B"), LOSS=("#EDE3F6","#6A3D9A"), UP=("#C8EAD3","#1E7A7A"),
           WHITE=("#FFFFFF","#BBBBBB"))
OUT = "/Volumes/2022-docs/科研工作2025/CCMP-2024/2025"

def b64(name):
    return "data:image/png;base64," + base64.b64encode(open(f"temp/figures/{name}","rb").read()).decode()

def build(cells, pw, ph, name, agent="SDA-Diff"):
    xml=f'''<mxfile host="app.diagrams.net" agent="{agent}" version="24.0.0"><diagram id="d" name="{name}"><mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{pw}" pageHeight="{ph}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{''.join(cells)}</root></mxGraphModel></diagram></mxfile>'''
    fn=f"{OUT}/{name}.drawio"
    open(fn,"w").write(xml)
    return fn

# =================== (1) CONTEXT ENCODER ===================
cells=[]; nid=2
def V(x,y,w,h,text,kind,bold=False,fs=11,fill=True,stroke_w=1.5,dashed=False):
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
def E(src,tgt,color="#444444",dashed=False,width=1.3):
    global nid
    st=f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeColor={color};strokeWidth={width};{('dashed=1;' if dashed else '')}"
    cells.append(f'<mxCell id="{nid}" value="" style="{st}" edge="1" parent="1" source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    nid+=1; return nid-1

# ---- context encoder schematic (canvas 1000x620) ----
V(20,15,300,50,"<b>c = [B, T=12, 6, H, W]</b>","DATA",bold=True,fs=12)
g1=V(20,95,190,55,"GESLA-3 + valid_mask","DATA",fs=10)
g2=V(225,95,190,55,"ERA5 msl, u10, v10","DATA",fs=10)
g3=V(430,95,190,55,"coarse GTSM","DATA",fs=10)
att=V(20,185,600,55,"<b>Channel-group temporal attention</b> over T=12 frames (att_group)","CONV",bold=True,fs=11)
zc=V(20,275,260,55,"context vector z_c","CONV",fs=11)
te=V(480,275,230,55,"<b>SinusoidalPosEmb</b><br>σ (diffusion time) + lead L","CONV",bold=True,fs=10)
mlp=V(20,365,420,60,"<b>per-UNet-level MLP</b> \u2192 (\u03b3_l, \u03b2_l)","CONV",bold=True,fs=11)
lv1=V(20,460,120,40,"level 1","CONV",fs=10); lv2=V(160,460,120,40,"level 2","CONV",fs=10); lv3=V(300,460,120,40,"level 3","CONV",fs=10)
# ResBlock application
rb=V(520,275,460,270,"","WHITE",dashed=True)
V(535,283,430,28,"<b>ResBlock at UNet level l</b>","CONV",bold=True,fs=11,dashed=True)
rb_h=V(545,325,120,55,"h<br>(features)","DATA",fs=10)
rb_c=V(545,400,120,60,"Conv + Act","CONV",fs=10)
rb_t=V(695,325,150,55,"time-FiLM:<br>h\u2190h\u00b7(1+\u03b3_t)+\u03b2_t","NOISE",fs=9)
rb_cf=V(695,400,150,55,"<b>context-FiLM:</b><br>h\u2190h\u00b7(1+\u03b3_l)+\u03b2_l","CONV",bold=True,fs=9)
rb_o=V(880,360,85,60,"out","DATA",fs=10)
E(cells and 0,0) if False else None
# edges (rebuild using ids captured)
# (handled below after ids assigned)
# wires:
E(g1,att,color="#2F7D46"); E(g2,att,color="#2F7D46"); E(g3,att,color="#2F7D46")
E(att,zc,color="#2F7D46"); E(te,mlp,color="#2F7D46",dashed=True)
E(zc,mlp,color="#2F7D46"); E(mlp,lv1,color="#2F7D46"); E(mlp,lv2,color="#2F7D46"); E(mlp,lv3,color="#2F7D46")
E(lv1,rb_cf,color="#2F7D46",dashed=True); E(lv2,rb_cf,color="#2F7D46",dashed=True); E(lv3,rb_cf,color="#2F7D46",dashed=True)
E(te,rb_t,color="#B06A1B",dashed=True)
E(rb_h,rb_c,color="#2F7D46"); E(rb_c,rb_t,color="#2F7D46"); E(rb_t,rb_cf,color="#2F7D46"); E(rb_cf,rb_o,color="#2F7D46")
fn1=build(cells,1000,620,"schematic_context_encoder")

# =================== (2) 2x SUPER-RESOLUTION ===================
cells=[]; nid=2
V(20,15,300,50,"<b>Input</b>","DATA",bold=True,fs=12)
IMG(25,85,150,150,"gulf_thumb_gtsm.png")
IMG(195,85,150,150,"gulf_thumb_sparse.png")
V(25,240,150,40,"coarse GTSM<br>(72 stations \u2248 50\u00d750)","DATA",fs=8)
V(195,240,150,40,"GESLA-3 obs<br>(sparse gauges)","DATA",fs=8)
ups=V(420,110,210,60,"<b>spatial upsampling</b><br>to 512\u00d7512 grid (shape only)","NOISE",bold=True,fs=9)
rd=V(700,70,300,140,"<b>Reverse diffusion at 512\u00d7512</b><br>\u03c3: 1.0 \u2192 0, 20 steps (EDM Heun)<br><font size=\"2\">generates fine structure from<br>the learned surge-field prior</font>","UP",bold=True,fs=10)
IMG(1060,70,230,230,"gulf_thumb_mean.png")
V(1060,305,230,35,"dense 512\u00d7512 surge field<br>(land-masked)","DATA",fs=8)
# comparison note
V(420,300,610,60,"<b>Bilinear upscale</b>: no new information (detail 0.006) &nbsp;&nbsp;|\u00a0\u00a0 <b>Generative SR</b>: 78\u00d7 more detail (0.46)","WHITE",bold=True,fs=10,dashed=True)
E(25+0,0,0) if False else None
# ids: c 2, gtsm-img 3, sparse-img 4, gtsm-lab 5, sparse-lab 6, ups 7, rd 8, out-img 9, out-lab 10, note 11
E(3,7,color="#B06A1B"); E(4,7,color="#B06A1B"); E(7,8,color="#1E7A7A"); E(8,9,color="#1E7A7A")
fn2=build(cells,1330,360,"schematic_super_resolution")

# ---- export both to PNG ----
DRAWIO="/Applications/draw.io.app/Contents/MacOS/draw.io"
for fn,name in [(fn1,"schematic_context_encoder"),(fn2,"schematic_super_resolution")]:
    r=subprocess.run([DRAWIO,"--export","--format","png","--output",f"{OUT}/{name}.png","--width",("1000" if "context" in name else "1330"),fn],capture_output=True,text=True)
    print(name, "drawio ok, export:", "OK" if r.returncode==0 else r.stderr[-200:])
    # copy to SDADiff
    import shutil
    shutil.copy(f"{OUT}/{name}.drawio","temp/figures/")
    shutil.copy(f"{OUT}/{name}.png","temp/figures/")
print("SCHEMATICS DONE")
