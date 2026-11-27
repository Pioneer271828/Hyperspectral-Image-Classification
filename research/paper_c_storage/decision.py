"""Render a single printable decision sheet from measured aggregate receipts."""
import argparse
import json
from pathlib import Path

import fitz

from .benchmark import clean
from .codec import require,hashed


def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);p.add_argument('--measurement',required=True);p.add_argument('--recovery',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();clean(a.expected_commit);measurement=Path(a.measurement);recovery=Path(a.recovery)
    x=json.loads(measurement.read_text());r=json.loads(recovery.read_text());out=Path(a.out);out.mkdir(exist_ok=False)
    require(x['source_states']==60 and r['new_synthetic_optimizer_updates']==10 and r['real_updates']==0,'Expected measured scope differs')
    doc=fitz.open();page=doc.new_page(width=595.276,height=841.89)
    # Existing PyMuPDF fallback; original full CJK face, no glyph-ID subsetting.
    font=fitz.Font('cjk');page.insert_font(fontname='decisionCJK',fontbuffer=font.buffer)
    navy=(.10,.18,.29);ink=(.13,.19,.26);muted=(.32,.39,.46)
    def text(rect,content,size=9.5,color=ink):
        remaining=page.insert_textbox(fitz.Rect(rect),content,fontname='decisionCJK',fontsize=size,lineheight=1.45,color=color)
        require(remaining>=0,'Decision text overflow')
    page.draw_rect(fitz.Rect(42,36,553,91),fill=navy,color=navy)
    text((54,45,541,72),'存储决策：保留逐步恢复，先解决容量',16,(1,1,1))
    text((54,72,541,88),'C14 / Issue 539   -   现有登记不改，真实训练仍暂停',8.5,(.85,.91,.96))
    text((44,106,551,169),'建议：不替换现有产物，也不降低留存频率。无损分块/差分备选已完成小规模恢复测试；其空间收益不足以解除当前LXC阻塞。未来如采用，须另立版本和预算。',10.4)
    text((44,179,551,214),'测量范围：C13已生成的60个合成状态，原始135.36MB；180个原文件哈希不变。整文件去重节省为0。未运行真实训练或query评分。',9.3)
    cols=[44,253,355,444,551];top=224;rh=31
    rows=[['完整恢复备选','关闭后大小','节省','本次测时'],
          ['64KiB分块 + 无损压缩','113.04MB','16.49%','9.22秒'],
          ['ZIP成员对齐 + 无损差分','104.66MB','22.68%','15.57秒']]
    for i,row in enumerate(rows):
        y=top+i*rh;page.draw_rect(fitz.Rect(cols[0],y,cols[-1],y+rh),fill=(.91,.94,.97) if i==0 else (1,1,1),color=(.78,.82,.86),width=.5)
        for j,cell in enumerate(row):text((cols[j]+6,y+8,cols[j+1]-4,y+rh-2),cell,8.6)
    text((44,330,551,350),'恢复与原子性',11,navy)
    text((44,352,551,424),'全部60个状态可还原原始完整文件SHA。差分最多2层，最多3次对象读回；本批逐字节验证耗时2.81秒，普通分块为1.47秒。九项测试覆盖强制结束进程、并发写入拒绝、提交原子性、篡改/截断和未完成恢复拒绝。四组随机断点的10次新合成更新，与原归档完整状态一致。',9.2)
    text((44,439,551,459),'36例空间情景及代价',11,navy)
    text((44,462,551,548),'四步早期合成轨迹外推：差分约189.30-212.43GB/副本，普通分块约203.69-205.76GB。双独立副本约379-425GB，另需日志与工作空间。编码/提交组件外推约3.22-6.24小时。\n这些是情景估算，不是真实训练工时或容量上限；长期熵、元数据增长、并发和目的地性能尚待验证。',9.2)
    text((44,565,551,585),'仍需批准的存储决定',11,navy)
    text((44,588,551,726),'原C13的897GB登记预留保持不变，新物理格式另定数据库、WAL、内存及备份硬预算。现有LXC仅余数GB，不能承载任一情景。\n\n建议先批准LXC可用的持久卷和独立备份位置；准确设备、挂载路径与容量尚未核实或获批。旧产物不删除、不搬走，历史写入器不改。新训练器接入、长序列压力测试与真实角色/预算均须另行登记。',9.4)
    page.draw_line((44,749),(551,749),color=(.76,.81,.86),width=.6)
    text((44,758,551,805),'大小含备选索引；原始参照为状态字节。本次测时含逐字节验证。\n测量生产版本：'+x['producer']+'\n实施、合同区分、实际I/O和恢复明细见随附源码及去身份回执。',7.3,muted)
    pdf=out/'storage-decision-one-page.pdf';doc.save(pdf,garbage=4,deflate=True);doc.close()
    with fitz.open(pdf) as check:
        require(len(check)==1,'Decision sheet is not one page')
        extracted=check[0].get_text();require('存储决策' in extracted and '212.43GB' in extracted,'Saved CJK/text extraction differs')
        check[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out/'storage-decision-preview.png')
        fonts=[list(row) for row in check[0].get_fonts(full=True)]
    (out/'render-receipt.json').write_text(json.dumps({'producer':a.expected_commit,'measurement_sha256':hashed(measurement.read_bytes()),'recovery_sha256':hashed(recovery.read_bytes()),'pdf_sha256':hashed(pdf.read_bytes()),'pages':1,'renderer':'existing PyMuPDF1.26.7; reportlab unavailable; full original embedded CJK face, no subsetting','fonts':fonts,'visual_review':'pending'},indent=2))
    print(json.dumps({'pdf_bytes':pdf.stat().st_size,'pages':1,'rendered':True}))


if __name__=='__main__':main()
