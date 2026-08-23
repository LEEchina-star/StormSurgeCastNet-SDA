# -*- coding: utf-8 -*-
"""SurgeCast framework v3: embed the two mechanism schematics at the bottom."""
import base64, re

SRC = "/Volumes/2022-docs/科研工作2025/CCMP-2024/2025/SurgeCast(1)_v2.drawio"
OUT = "/Volumes/2022-docs/科研工作2025/CCMP-2024/2025/SurgeCast(1)_v3.drawio"

def b64(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()

xml = open(SRC, encoding="utf-8").read()
# expand page height 1030 -> 1400
xml = xml.replace('pageHeight="1030"', 'pageHeight="1400"')
# add schematic cells before </root>
ctx_img = b64("temp/figures/schematic_context_encoder.png")
sr_img = b64("temp/figures/schematic_super_resolution.png")
extra = f'''
<mxCell id="s1" value="<b>Mechanism details</b>" style="rounded=0;whiteSpace=wrap;html=1;fontSize=14;fontStyle=1;fontColor=#2F7D46;align=left;" vertex="1" parent="1"><mxGeometry x="560" y="1010" width="1400" height="30" as="geometry"/></mxCell>
<mxCell id="s2" value="" style="shape=image;aspect=fixed;shadow=1;html=1;image={ctx_img};" vertex="1" parent="1"><mxGeometry x="560" y="1050" width="520" height="292" as="geometry"/></mxCell>
<mxCell id="s3" value="" style="shape=image;aspect=fixed;shadow=1;html=1;image={sr_img};" vertex="1" parent="1"><mxGeometry x="1120" y="1050" width="700" height="197" as="geometry"/></mxCell>
'''
xml = xml.replace('</root>', extra + '</root>')
open(OUT, "w", encoding="utf-8").write(xml)
print("v3 saved:", OUT)
