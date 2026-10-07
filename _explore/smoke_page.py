"""页面冒烟测试：抓取线上页面 → 校验 JS 语法 → 用真实数据跑一遍渲染。

为什么要这个：模板里改 JS 很容易写出「语法没错但渲染出 undefined / 行数不对」的问题，
而浏览器起不来（Windows 上 agent-browser 不可用），肉眼也看不全 6 个板块 × 多种视图。
这个脚本三步走：
  ① 抽出页面的 <script> 段，用 node --check 查语法
  ② 抽出页面里内联的 DATA / SUM（真实数据）
  ③ 在 node 里 stub 掉 document，真实调用 drawChart()/render()，校验点数/末点坐标/等级标记/NaN

用法：
  python _explore/smoke_page.py                # 默认 http://127.0.0.1:8779/
  python _explore/smoke_page.py --port 8780

⚠️ 2026-09-23 起「合并」视图 = **三个模型各一张图**（drawUnion），不再叠三条曲线。
   所以合并视图的校验改成「3 张子图、各自主折线/触发线/信号点」，
   单模型视图（跌得深 / 跌势放缓 / 均线止跌）才用原来的单折线校验。
⚠️ 2026-10-07 界面改版（大白话仪表盘）：左侧分值卡/刻度条/最近信号行已移除，
   改为「今日结论横幅 + 6 板块卡（三模型温度上卡）+ 模型 Tab」；
   信号点统一由 sigMarker() 生成（class="sigmk" / 光环 class="halo"），按 class 计数。
"""
import sys, os, re, json, subprocess, argparse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = r"C:/Users/wr/.workbuddy-ai/binaries/node/versions/22.22.2-3/node.exe"
TMP = os.path.join(HERE, "_smoke_tmp")
os.makedirs(TMP, exist_ok=True)


JS_STUB = r"""
/* 最小 DOM stub：够跑通页面脚本（含 tab/范围按钮创建、drawChart 写 svg） */
const _els={};
function _mk(id){return {id:id,checked:false,innerHTML:'',textContent:'',dataset:{},style:{},
  clientWidth:900,className:'',classList:{toggle:function(){},add:function(){},remove:function(){}},
  appendChild:function(){},querySelector:function(){return _mk('svg');},
  querySelectorAll:function(){return [];},addEventListener:function(){},
  setAttribute:function(){},getBoundingClientRect:function(){return {left:0,width:900};}};}
global.document={getElementById:function(id){return _els[id]||(_els[id]=_mk(id));},
                 createElement:function(){return _mk('new');},
                 querySelector:function(){return _mk('q');},
                 querySelectorAll:function(){return [];}};
global.window={addEventListener:function(){}};
"""

JS_CHART = r"""
/* 图表冒烟：直接跑 drawChart()，检查切片点数、末点坐标、NaN/undefined、口径提示 */
let bad2=0;
const RD={'7d':7,'1m':30,'3m':91,'6m':182,'1y':365};
function expectN(dates,rg){
  const days=RD[rg];
  if(days==null) return dates.length;
  const t=new Date(dates[dates.length-1]+'T00:00:00'); t.setDate(t.getDate()-days);
  const cut=t.getFullYear()+'-'+String(t.getMonth()+1).padStart(2,'0')
           +'-'+String(t.getDate()).padStart(2,'0');
  let c=0; for(const d of dates) if(d>=cut) c++;
  return Math.min(Math.max(c,5), dates.length);
}
const MAIN_RE=/<path d="([^"]+)" fill="none" stroke="#63B3FF" stroke-width="2\.2"/;
const cnt=(s,re)=>(s.match(re)||[]).length;
const COL3={swing:'#63B3FF',swing2:'#C9A2FF',swing3:'#FFB86B'};

/* ---------- ② 大机会 SENTI-1 ---------- */
MODE='senti';
for(const k of Object.keys(DATA)){
  cur=k;
  const line=[];
  for(const rg of ['7d','1m','3m','6m','1y','all']){
    RNG=rg; drawChart();
    const h=document.getElementById('plot').innerHTML||'';
    const info=document.getElementById('rnginfo').innerHTML||'';
    if(/undefined|NaN/.test(h+info)){console.log('  !! '+k+' '+rg+' 输出含 undefined/NaN');bad2++;continue;}
    const m=h.match(MAIN_RE);
    if(!m){console.log('  !! '+k+' '+rg+' 找不到主折线');bad2++;continue;}
    const pts=(m[1].match(/L/g)||[]).length+1;
    const exp=expectN(DATA[k].dates,rg);
    const lastX=parseFloat(m[1].slice(m[1].lastIndexOf('L')+1).trim().split(' ')[0]);
    const okP=(pts===exp), okX=(Math.abs(lastX-782)<0.6);   // 末点应贴住绘图区右边界 62+720
    line.push(rg+':'+pts+(okP?'':'(应'+exp+')')+(okX?'':'(末点x='+lastX+')'));
    if(!okP||!okX) bad2++;
    /* 等级标记：sigmk 三角形数 == 窗口内信号数；halo 光环数 == 窗口内 A 级数 */
    const [w0,w1]=winIdx(DATA[k]);
    const d0=DATA[k].dates[w0], d1=DATA[k].dates[w1];
    const inW=DATA[k].bottom.filter(e=>e.date>=d0&&e.date<=d1);
    const nTri=cnt(h,/class="sigmk"/g);
    const nHalo=cnt(h,/class="halo"/g);
    const nA=inW.filter(e=>e.grade==='A').length;
    if(nTri!==inW.length){console.log('  !! '+k+' '+rg+' 信号标记 '+nTri+' != 窗口信号 '+inW.length);bad2++;}
    if(nHalo!==nA){console.log('  !! '+k+' '+rg+' A级光环 '+nHalo+' != A级信号 '+nA);bad2++;}
    if(!info.includes(DATA[k].dates[DATA[k].dates.length-1])){
      console.log('  !! '+k+' '+rg+' 窗口信息缺少末日');bad2++;}
  }
  console.log('  '+k.padEnd(9)+line.join('  '));
}
/* 首屏三要素：今日结论横幅 + 6 板块卡 + 大白话说明，任何板块选中时都必须正常渲染 */
let badLs=0;
for(const k of Object.keys(DATA)){
  cur=k; render();
  const bn=document.getElementById('banner').innerHTML||'';
  const bd=document.getElementById('boards').innerHTML||'';
  const gb=document.getElementById('gbody').innerHTML||'';
  if(!/扫描结果/.test(bn)||/undefined|NaN/.test(bn)){
    console.log('  !! '+k+' 今日结论横幅异常: '+bn.slice(0,90));badLs++;}
  if(cnt(bd,/class="bd/g)!==6||/undefined|NaN/.test(bd)){
    console.log('  !! '+k+' 板块卡数量异常或含 undefined/NaN');badLs++;}
  if(/undefined|NaN/.test(gb)){console.log('  !! '+k+' 大白话说明含 undefined/NaN');badLs++;}
}
if(badLs===0) console.log('  横幅/板块卡/说明: 6 个板块均已渲染 ✔');
bad2+=badLs;
console.log('\n  '+(bad2===0?'✔ SENTI-1 冒烟通过':'✘ SENTI-1 存在 '+bad2+' 处问题'));

/* ---------- ③ 小波段单模型视图（回调底 SWING） ---------- */
console.log('\n③ 小波段 SWING 单模型视图（回调底）');
let bad3=0;
MODE='swing'; SRC='swing';
if(!(SUM.swing&&SUM.swing.total)){console.log('  !! SUM.swing 缺失');bad3++;}
for(const k of Object.keys(DATA)){
  cur=k;
  const line=[];
  for(const rg of ['7d','1m','3m','6m','1y','all']){
    RNG=rg; drawChart();
    const h=document.getElementById('plot').innerHTML||'';
    const info=document.getElementById('rnginfo').innerHTML||'';
    if(/undefined|NaN/.test(h+info)){console.log('  !! '+k+' '+rg+' 输出含 undefined/NaN');bad3++;continue;}
    if(cnt(h,/<svg /g)!==1){console.log('  !! '+k+' '+rg+' 单模型视图应有 1 张图');bad3++;continue;}
    const m=h.match(MAIN_RE);
    if(!m){console.log('  !! '+k+' '+rg+' 找不到主折线');bad3++;continue;}
    const pts=(m[1].match(/L/g)||[]).length+1;
    const exp=expectN(DATA[k].dates,rg);
    if(pts!==exp){console.log('  !! '+k+' '+rg+' 点数 '+pts+' != '+exp);bad3++;}
    line.push(rg+':'+pts);
  }
  render();
  const bn=document.getElementById('banner').innerHTML||'';
  const ms=document.getElementById('modestat').innerHTML||'';
  const lg=document.getElementById('legend').innerHTML||'';
  const gb=document.getElementById('gbody').innerHTML||'';
  if(/undefined|NaN/.test(bn+ms+lg+gb)){console.log('  !! '+k+' SWING 文案含 undefined/NaN');bad3++;}
  if(!/跌得深/.test(lg)){console.log('  !! '+k+' 图例未切到小波段（跌得深）');bad3++;}
  console.log('  '+k.padEnd(9)+line.join('  '));
}
/* 目标日期硬约束：科创板必须能标出 2026-08-03 与 2026-09-14 */
{
  const ev=(DATA.STAR.swing_events||[]).map(e=>e.date);
  for(const d of ['2026-08-03','2026-09-14']){
    const ok=ev.includes(d);
    console.log('  科创板 '+d+' 标记: '+(ok?'✔':'✘ 未触发'));
    if(!ok) bad3++;
  }
}
console.log('\n  '+(bad3===0?'✔ SWING 单模型冒烟通过':'✘ SWING 单模型存在 '+bad3+' 处问题'));

/* ---------- ④ 模型 Tab 四选：三个一起看（三张图）/ 跌得深 / 跌势放缓 / 均线止跌 ---------- */
console.log('\n④ 小波段模型 Tab 四选（三个一起看 / 跌得深 / 跌势放缓 / 均线止跌）');
let bad4=0;
for(const s of ['union','swing','swing2','swing3']){
  SRC=s;
  let n=0, bad=0;
  for(const k of Object.keys(DATA)){
    cur=k; RNG='all'; drawChart(); render();
    const h=document.getElementById('plot').innerHTML||'';
    const ms=document.getElementById('modestat').innerHTML||'';
    const lg=document.getElementById('legend').innerHTML||'';
    const bd=document.getElementById('boards').innerHTML||'';
    const info=document.getElementById('rnginfo').innerHTML||'';
    if(/undefined|NaN/.test(h+ms+lg+bd+info)){console.log('  !! '+s+' '+k+' 含 undefined/NaN');bad++;}
    if(s==='union'){
      /* 三张子图，各自一条主折线 + 一条触发线 + 自己的信号点 */
      const nsvg=cnt(h,/<svg /g);
      if(nsvg!==3){console.log('  !! union '+k+' 子图数 '+nsvg+' != 3');bad++;}
      if(cnt(h,/class="subchart"/g)!==3){console.log('  !! union '+k+' 缺 subchart 容器');bad++;}
      if(cnt(h,/（跌破就提示）/g)!==3){console.log('  !! union '+k+' 触发线数 '+cnt(h,/（跌破就提示）/g)+' != 3');bad++;}
      for(const key of ['swing','swing2','swing3']){
        const re=new RegExp('stroke="'+COL3[key]+'" stroke-width="1\\.8"');
        if(!re.test(h)){console.log('  !! union '+k+' 缺 '+key+' 主折线');bad++;}
      }
      /* 2026-10-07 起小机会模式 rnginfo/modestat 与「三个模型各画一张图」图例条已移除（用户要求精简） */
      if(/三个模型各自一张图|三个模型各画一张图/.test(info+lg)){console.log('  !! union '+k+' 已移除的口径提示又出现了');bad++;}
      if(/闭眼买/.test(ms)){console.log('  !! union '+k+' 小机会 modestat 应已移除');bad++;}
      /* 信号点：src='both' 的事件三张图都画 → 期望 sigmk 数 = Σ(1 or 3) */
      const [w0,w1]=winIdx(DATA[k]);
      const d0=DATA[k].dates[w0], d1=DATA[k].dates[w1];
      const inW=(DATA[k].union_events||[]).filter(e=>e.date>=d0&&e.date<=d1);
      const expTri=inW.reduce((a,e)=>a+((e.src==='both')?3:1),0);
      const nTri=cnt(h,/class="sigmk"/g);
      if(nTri!==expTri){console.log('  !! union '+k+' 信号点 '+nTri+' != 期望 '+expTri);bad++;}
      /* 板块卡：三个模型的最新值都要上卡 */
      if(!/跌得深/.test(bd)||!/跌势放缓/.test(bd)||!/均线止跌/.test(bd)){
        console.log('  !! union '+k+' 板块卡缺某个模型的最新值');bad++;}
      n+=inW.length;
    }else{
      if(cnt(h,/<svg /g)!==1){console.log('  !! '+s+' '+k+' 单模型视图应有 1 张图');bad++;}
      const re1=new RegExp('stroke="'+COL3[s]+'" stroke-width="2\\.2"');
      if(!re1.test(h)){console.log('  !! '+s+' '+k+' 找不到主折线');bad++;}
      n+=(DATA[k][s+'_events']||[]).length;
    }
  }
  console.log('  SRC='+s.padEnd(7)+' 6板块信号合计 '+String(n).padStart(4)
              +(bad?('  ✘ '+bad+' 处问题'):'  ✔'));
  bad4+=bad;
}
SRC='union';
/* 大白话说明必须同时讲清三个模型（用户反馈「看不懂」才改成大白话的，别被后续改动删掉） */
MODE='swing'; render();
const shTxt=document.getElementById('gbody').innerHTML||'';
if(!/跌得深/.test(shTxt)||!/跌势放缓/.test(shTxt)||!/均线止跌/.test(shTxt)){
  console.log('  !! 大白话说明缺少三个模型的解释（需同时含「跌得深」「跌势放缓」「均线止跌」）');bad4++;}
console.log('\n  '+(bad4===0?'✔ 模型 Tab 四选冒烟通过':'✘ 模型 Tab 四选存在 '+bad4+' 处问题'));

/* ---------- ⑤ 盘中刷新：盘中区常显；数值只在 showUntil 前显示，其余一律「—」 ---------- */
console.log('\n⑤ 盘中刷新按钮与盘中区');
let bad5=0;
MODE='swing'; SRC='union'; INTRA=null; render();
let bd5=document.getElementById('boards').innerHTML||'';
if(cnt(bd5,/class="intr"/g)!==6){console.log('  !! 未刷新时盘中区也应常显（6 卡）');bad5++;}
if(!/盘中（未刷新）/.test(bd5)||!/—/.test(bd5)){console.log('  !! 未刷新时应显示占位符 —');bad5++;}
/* 交易时段点击（showUntil=未来）→ 显示数值 */
INTRA={ts:'10-07 14:32',live:true,clickedDay:todayStr(),showUntil:Date.now()+3600e3,boards:{}};
for(const k of Object.keys(DATA)) INTRA.boards[k]={senti:21.8,swing:62.3,swing2:38.9,swing3:23.0};
render();
bd5=document.getElementById('boards').innerHTML||'';
if(cnt(bd5,/class="intr"/g)!==6){console.log('  !! 小机会模式盘中区数 '+cnt(bd5,/class="intr"/g)+' != 6');bad5++;}
if(!/盘中 10-07 14:32/.test(bd5)){console.log('  !! 盘中区缺时间戳');bad5++;}
if(!/62.3/.test(bd5)||!/38.9/.test(bd5)||!/23.0/.test(bd5)){console.log('  !! 小机会盘中区缺某个模型值');bad5++;}
if(/undefined|NaN/.test(bd5)){console.log('  !! 盘中区含 undefined/NaN');bad5++;}
MODE='senti'; render();
bd5=document.getElementById('boards').innerHTML||'';
if(cnt(bd5,/class="intr"/g)!==6){console.log('  !! 大机会模式盘中区数 '+cnt(bd5,/class="intr"/g)+' != 6');bad5++;}
if(!/21.8/.test(bd5)){console.log('  !! 大机会盘中区缺 SENTI-1 值');bad5++;}
/* 收盘后点击（live=false → showUntil=0）→ 显示「—」并标「已收盘」 */
MODE='swing'; INTRA.showUntil=0; render();
bd5=document.getElementById('boards').innerHTML||'';
if(/62.3/.test(bd5)){console.log('  !! 收盘后点击盘中值应显示 —');bad5++;}
if(!/盘中（已收盘）/.test(bd5)){console.log('  !! 收盘后点击应标「已收盘」');bad5++;}
/* 跨天（showUntil 已过期、clickedDay 是昨天）→ 回到「—」未刷新态 */
INTRA.showUntil=Date.now()-1; INTRA.clickedDay='2000-01-01'; render();
bd5=document.getElementById('boards').innerHTML||'';
if(/62.3/.test(bd5)){console.log('  !! 跨天后盘中值应清空为 —');bad5++;}
if(!/盘中（未刷新）/.test(bd5)){console.log('  !! 跨天后盘中区标题应回到未刷新态');bad5++;}
INTRA=null;
console.log('  '+(bad5===0?'✔ 盘中刷新冒烟通过':'✘ 盘中刷新存在 '+bad5+' 处问题'));
process.exit((bad2+bad3+bad4+bad5)?1:0);
"""


def fetch(url):
    req = urllib.request.Request(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=60).read().decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8779")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    url = f"http://{a.host}:{a.port}/"

    print("=" * 78)
    print("页面冒烟测试 · " + url)
    print("=" * 78)
    try:
        html = fetch(url)
    except Exception as e:
        print(f"  ✘ 抓不到页面：{e}"); return 2

    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    if not scripts:
        print("  ✘ 页面里没有 <script> 段"); return 2
    if 'id="intrabtn"' not in html:
        print("  ✘ 页面缺少「盘中刷新」按钮（id=intrabtn）"); return 1
    js = scripts[-1]
    js_path = os.path.join(TMP, "page.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js)

    print("\n① JS 语法检查（node --check）")
    r = subprocess.run([NODE, "--check", js_path], capture_output=True, text=True)
    if r.returncode != 0:
        print("  ✘ 语法错误：\n" + (r.stderr or "")[:2000]); return 1
    print(f"  ✔ 通过（{len(js)} 字符）")

    m1 = re.search(r"const DATA\s*=\s*(\{.*?\});\s*\n", html, re.S)
    m2 = re.search(r"const SUM\s*=\s*(\{.*?\});", html, re.S)
    if not (m1 and m2):
        print("  ✘ 页面里找不到内联的 DATA / SUM，没法做真实数据渲染"); return 2
    d_path = os.path.join(TMP, "data.json")
    with open(d_path, "w", encoding="utf-8") as f:
        json.dump({"DATA": json.loads(m1.group(1)), "SUM": json.loads(m2.group(1))},
                  f, ensure_ascii=False)

    print("\n② 用真实数据渲染图表（6 板块 × 6 个时间范围，直接跑 drawChart）")
    c_path = os.path.join(TMP, "chart.js")
    with open(c_path, "w", encoding="utf-8") as f:
        f.write(JS_STUB + "\n" + js + "\n" + JS_CHART)
    r = subprocess.run([NODE, c_path], capture_output=True, text=True)
    out = (r.stdout or "").rstrip()
    print(out)
    if r.returncode != 0 or "<<<" in out:
        print((r.stderr or "")[:1500])
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
