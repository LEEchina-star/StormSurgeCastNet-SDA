# -*- coding: utf-8 -*-
"""draw.io v4: 3D-shadow boxes + real-data thumbnails + PROCESS diagrams
(denoising & SDA assimilation shown with real surge fields)."""
import html, base64

PAL = dict(DATA=("#DCE9F7","#3E6B9E"),CONV=("#E3F0E6","#2F7D46"),NOISE=("#F5E6D3","#B06A1B"),
           DA=("#FBE3DE","#C0392B"),LOSS=("#EDE3F6","#6A3D9A"),UP=("#C8EAD3","#1E7A7A"),WHITE=("#FFFFFF","#BBBBBB"))

def b64(name):
    return "data:image/png;base64," + base64.b64encode(open(f"temp/figures/{name}","rb").read()).decode()

cells=[]; nid=2
def V(x,y,w,h,text,kind,bold=False,fs=13,fill=True,tc=None,stroke_w=1.5):
    global nid
    f,s=PAL[kind]
    st=(f"rounded=1;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};strokeWidth={stroke_w};"
        f"fontSize={fs};fontColor={'#222222' if tc is None else tc};fontStyle={'1' if bold else '0'};"
        f"shadow=1;gradientColor=#FFFFFF;gradientDirection=north;{'fillColor=none;' if not fill else ''}")
    cells.append(f'<mxCell id="{nid}" value="{html.escape(text)}" style="{st}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    nid+=1; return nid-1
def IMG(x,y,w,h,name):
    global nid
    cells.append(f'<mxCell id="{nid}" value="" style="shape=image;aspect=fixed;shadow=1;html=1;image={b64(name)};" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    nid+=1; return nid-1
def E(src,tgt,color="#444444",dashed=0,width=1.4):
    global nid
    st=f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeColor={color};strokeWidth={width};{('dashed=1;' if dashed else '')}"
    cells.append(f'<mxCell id="{nid}" value="" style="{st}" edge="1" parent="1" source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    nid+=1; return nid-1

# ===== INPUT =====
V(40,20,340,34,"Input","DATA",bold=True,fs=15)
geo=IMG(90,60,240,270,"geo_fujian.png")   # Fujian coast + ROI (geographic map)
i1=IMG(50,345,140,140,"thumb_sparse.png"); i2=IMG(200,345,140,140,"thumb_era5_msl.png")
i3=IMG(50,495,140,140,"thumb_wind.png"); i4=IMG(200,495,140,140,"thumb_gtsm.png")
cbox=V(40,650,340,44,"<b>c=[B,T,6,H,W]</b> + lead L","DATA",bold=True,fs=12)
for t in [i1,i2,i3,i4]: E(t,cbox,color="#3E6B9E",width=1.1)

# ===== MODEL =====
V(420,20,900,480,"","WHITE",fill=False,stroke_w=1.0)
V(420,20,900,34,"Model — EDM generation + SDA assimilation","CONV",bold=True,fs=15)

# training (upper) — with denoising process diagram
V(440,70,860,190,"","WHITE",fill=False)
V(455,78,830,30,"Training — learn to recover the surge field from noise","CONV",bold=True,fs=12,fill=False,tc="#2F7D46")
x0=V(455,130,90,60,"x\u2080","DATA",fs=14)
nz=V(575,130,130,60,"x\u209c=x\u2080+\u03c3\u03b5","NOISE",fs=12)
E(x0,nz,color="#B06A1B")
pden=IMG(735,120,430,150,"proc_denoise.png")   # denoising process (clean->noisy->denoised)
E(nz,pden)
xhat=V(1180,130,110,60,"x\u0302\u2080","CONV",fs=14)
E(pden,xhat)
loss=V(455,270,835,40,"<b>L = E[\u03bb(\u03c3)\u2016D\u03b8(x\u2080+\u03c3\u03b5)\u2212x\u2080\u2016\u00b2]</b>  (masked)","LOSS",bold=True,fs=12)
E(pden,loss,color="#6A3D9A",dashed=1)

# inference (lower) — with SDA assimilation process diagram
V(440,340,860,155,"","WHITE",fill=False)
V(455,348,830,30,"Inference — correct the sampling path with new observations","DA",bold=True,fs=12,fill=False,tc="#C0392B")
nx=V(455,410,80,56,"noise","DATA",fs=12)
passim=IMG(565,360,430,160,"proc_assim.png")   # assimilation process (prior->obs->posterior)
E(nx,passim)
sda=V(1025,410,260,56,"<b>\u2207log p(x\u209c|y,c)=s\u03b8+\u2207log N(y|Ax\u0302\u2080,R)</b>","DA",bold=True,fs=11,tc="#C0392B",stroke_w=2.2)
E(passim,sda,color="#C0392B")

# ===== OUTPUT =====
V(1360,20,320,480,"","WHITE",fill=False)
V(1360,20,320,34,"Output","DATA",bold=True,fs=15)
o1=IMG(1370,70,150,150,"thumb_out_mean.png"); o2=IMG(1525,70,150,150,"thumb_out_interval.png")
o3=IMG(1370,235,150,150,"thumb_out_exceed.png")
V(1370,395,150,26,"mean","UP",fs=10)
V(1525,395,150,26,"90% interval","UP",fs=10)
V(1370,428,305,26,"P(surge>h)","UP",fs=10)
E(cbox,pden,color="#2F7D46")
E(sda,o1,color="#C0392B")

xml=f'''<mxfile host="app.diagrams.net" agent="SDA-Diff" version="24.0.0"><diagram id="sdadiff" name="SDA-Diff framework v4"><mxGraphModel dx="1800" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1720" pageHeight="730" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{''.join(cells)}</root></mxGraphModel></diagram></mxfile>'''
open("temp/figures/fig1_sdadiff_v4.drawio","w").write(xml)
print(f"v4 生成：{len(cells)} 元素")
