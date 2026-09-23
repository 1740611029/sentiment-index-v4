# -*- coding: utf-8 -*-
"""把 index.html 的「合并视图」收尾：删掉叠三条曲线的死代码 + 改写图例。
每个替换都断言「恰好命中 1 次」，避免静默改错。"""
import io, sys, os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "web", "templates", "index.html")

src = io.open(P, encoding="utf-8").read()
orig_len = len(src)

R = []

# ---- 1. 去掉 sc2 / sc3 常量声明 ----
R.append((
r'''  const sc=SR(B).slice(i0,i1+1), cl=B.close.slice(i0,i1+1), dt=B.dates.slice(i0,i1+1);
  const sc2=null, sc3=null;
  const SWG = MODE==='swing';''',
r'''  const sc=SR(B).slice(i0,i1+1), cl=B.close.slice(i0,i1+1), dt=B.dates.slice(i0,i1+1);
  const SWG = MODE==='swing';'''))

# ---- 2. y 轴不再并两条别的曲线 ----
R.append((
r'''  let lo=Math.min(...sc), hi=Math.max(...sc);
  if(sc2){ lo=Math.min(lo,Math.min(...sc2)); hi=Math.max(hi,Math.max(...sc2)); }
  if(sc3){ lo=Math.min(lo,Math.min(...sc3)); hi=Math.max(hi,Math.max(...sc3)); }''',
r'''  let lo=Math.min(...sc), hi=Math.max(...sc);'''))

# ---- 3. 删掉「叠加 SWING-2 / SWING-3 曲线」两整段（合并视图已改三张图） ----
R.append((
r'''  // 合并视图：叠加 SWING-2 曲线（虚线）+ 它自己的阈值线
  if(sc2){
    let d2='';
    for(let a=i0;a<=i1;a++) d2+=(a>i0?'L':'M')+x(a).toFixed(1)+' '+y(sc2[a-i0]).toFixed(1);
    P.push(`<path d="${d2}" fill="none" stroke="#C9A2FF" stroke-width="1.6"
             stroke-dasharray="5 4" opacity=".85" stroke-linejoin="round"/>`);
    const T1=THR2();
    if(T1!=null&&T1>=yMin&&T1<=yMax){
      P.push(`<line x1="${L}" y1="${y(T1)}" x2="${L+PW}" y2="${y(T1)}" stroke="#C9A2FF"
               stroke-width="1.1" stroke-dasharray="3 4" opacity=".6"/>`);
      P.push(`<text x="${L+PW-4}" y="${y(T1)+12}" fill="#C9A2FF" font-size="10"
               text-anchor="end" opacity=".85">SWING-2 阈值 ${T1}</text>`);
    }
  }
  // 合并视图：再叠加 SWING-3 曲线（点线）+ 它的阈值线
  if(sc3){
    let d3='';
    for(let a=i0;a<=i1;a++) d3+=(a>i0?'L':'M')+x(a).toFixed(1)+' '+y(sc3[a-i0]).toFixed(1);
    P.push(`<path d="${d3}" fill="none" stroke="#FFB86B" stroke-width="1.5"
             stroke-dasharray="1.5 3.5" opacity=".85" stroke-linejoin="round"/>`);
    const T2=THR3();
    if(T2!=null&&T2>=yMin&&T2<=yMax){
      P.push(`<line x1="${L}" y1="${y(T2)}" x2="${L+PW}" y2="${y(T2)}" stroke="#FFB86B"
               stroke-width="1.1" stroke-dasharray="3 4" opacity=".6"/>`);
      P.push(`<text x="${L+6}" y="${y(T2)-5}" fill="#FFB86B" font-size="10"
               opacity=".85">SWING-3 阈值 ${T2}（须回升）</text>`);
    }
  }
''', ''))

# ---- 4. 信号点不再挑曲线 ----
R.append((
r'''    const g=gradeOf(e), col=GCOL[g]||GCOL.C;
    /* 合并视图下，各信号点挂到各自模型的曲线上（回调底→主曲线 / 动量拐点→SWING-2 / 均线拐点→SWING-3） */
    const useS=(sc2&&e.src==='macd')?sc2:((sc3&&e.src==='maslope')?sc3:sc);
    const cx=x(a).toFixed(1), cyy=y(useS[a-i0]), cyc=cyy+12;''',
r'''    const g=gradeOf(e), col=GCOL[g]||GCOL.C;
    const cx=x(a).toFixed(1), cyy=y(sc[a-i0]), cyc=cyy+12;'''))

# ---- 5. 信号点形状统一成三角（单模型视图） ----
R.append((
r'''    const fill=g==='C'?'none':col, sw=g==='C'?1.8:1.2;
    if(sc2&&e.src==='macd')
      P.push(`<path d="M${cx} ${(cyc-7).toFixed(1)} l6.5 7 l-6.5 7 l-6.5 -7 Z"
               fill="${fill}" stroke="${col}" stroke-width="${sw}" stroke-linejoin="round"/>`);
    else if(sc3&&e.src==='maslope')
      P.push(`<circle cx="${cx}" cy="${cyc.toFixed(1)}" r="5.5"
               fill="${fill}" stroke="${col}" stroke-width="${sw}"/>`);
    else{
      const ap=(cyc-8).toFixed(1), bs=(cyc+5).toFixed(1);
      P.push(`<path d="M${cx} ${ap} l-6.5 ${(bs-ap)} l13 0 Z"
               fill="${fill}" stroke="${col}" stroke-width="${sw}" stroke-linejoin="round"/>`);
    }''',
r'''    const fill=g==='C'?'none':col, sw=g==='C'?1.8:1.2;
    const ap=(cyc-8).toFixed(1), bs=(cyc+5).toFixed(1);
    P.push(`<path d="M${cx} ${ap} l-6.5 ${(bs-ap)} l13 0 Z"
             fill="${fill}" stroke="${col}" stroke-width="${sw}" stroke-linejoin="round"/>`);'''))

# ---- 6. 悬停几何里去掉 sc2 / sc3 ----
R.append((
r'''  el._geo={x,y,sc,sc2,sc3,close:cl,dates:dt,i0,N,W,H,L,PW,TOPY,BOTY,evmap,hold:HOLD};''',
r'''  el._geo={x,y,sc,close:cl,dates:dt,i0,N,W,H,L,PW,TOPY,BOTY,evmap,hold:HOLD};'''))

# ---- 7. 悬停标签去掉 union 分支（合并视图由 drawUnion 自己接管） ----
R.append((
r'''    const lbl = MODE!=='swing' ? `情绪分值 <b>${fmt(g.sc[i],1)}</b>`
      : SRC==='union'  ? `回调底 <b>${fmt(g.sc[i],1)}</b>　`
                        +`动量拐点 <b style="color:#C9A2FF">${fmt((g.sc2||[])[i],1)}</b>　`
                        +`均线拐点 <b style="color:#FFB86B">${fmt((g.sc3||[])[i],1)}</b>`
      : SRC==='swing2' ? `动量拐点 <b>${fmt(g.sc[i],1)}</b>`''',
r'''    const lbl = MODE!=='swing' ? `情绪分值 <b>${fmt(g.sc[i],1)}</b>`
      : SRC==='swing2' ? `动量拐点 <b>${fmt(g.sc[i],1)}</b>`'''))

# ---- 8. 合并视图图例改写 ----
R.append((
r'''const LEGEND_UNION =
  `<span><i style="background:var(--blu)"></i>回调底＝跌到哪了（SWING 实线，≤22 触发）</span>
   <span><i style="background:#C9A2FF"></i>动量拐点＝跌得多快（SWING-2 虚线）</span>
   <span><i style="background:#FFB86B"></i>均线拐点＝均线还跌得多急（SWING-3 点线，≤10 且须回升）</span>
   <span><i style="background:#5C6E85"></i>指数收盘（右轴）</span>
   <span><i style="background:var(--grn)"></i>≤ 阈值 入场区</span>
   <span><i style="background:var(--yel)"></i>▲＝回调底　◆＝动量拐点　●＝均线拐点</span>
   <span style="white-space:nowrap">等级＝当天有几个板块出信号（并集口径）：
     <b style="color:#4FD1E8">▲</b>S ≥5　
     <b style="color:#6FE39C">▲</b>A ≥4　
     <b style="color:#F5B544">▲</b>B 2~3　
     <b style="color:#96A5BA">△</b>C 仅自身</span>`;''',
r'''const LEGEND_UNION =
  `<span style="color:var(--yel)"><b>三个模型各画一张图</b>（刻度不同，不要横着比高低）</span>
   <span><i style="background:var(--blu)"></i>回调底 SWING　≤22 触发</span>
   <span><i style="background:#C9A2FF"></i>动量拐点 SWING-2　≤12 触发</span>
   <span><i style="background:#FFB86B"></i>均线拐点 SWING-3　≤10 且当天须回升</span>
   <span><i style="background:#5C6E85"></i>指数收盘（右轴）</span>
   <span><i style="background:var(--grn)"></i>≤ 各自阈值＝入场区</span>
   <span><i style="background:var(--yel)"></i>信号点</span>
   <span style="white-space:nowrap">等级＝当天有几个板块出信号（并集口径）：
     <b style="color:#4FD1E8">▲</b>S ≥5　
     <b style="color:#6FE39C">▲</b>A ≥4　
     <b style="color:#F5B544">▲</b>B 2~3　
     <b style="color:#96A5BA">△</b>C 仅自身</span>`;'''))

fail = []
for i, (o, n) in enumerate(R, 1):
    c = src.count(o)
    if c != 1:
        fail.append("R%d 命中 %d 次" % (i, c))
        continue
    src = src.replace(o, n)

if fail:
    print("ABORT:", "; ".join(fail))
    sys.exit(1)

io.open(P, "w", encoding="utf-8", newline="\n").write(src)
print("OK  8/8 替换完成  %d -> %d 字节" % (orig_len, len(src)))
