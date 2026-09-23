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
   所以合并视图的校验改成「3 张子图、各自主折线/阈值线/信号点」，
   单模型视图（回调底 / 动量拐点 / 均线拐点）才用原来的单折线校验。
"""
import sys, os, re, json, subprocess, argparse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = r"C:/Users/wr/.workbuddy-ai/binaries/node/versions/22.22.2-2/node.exe"
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
const MAIN_RE=/<path d="([^"]+)" fill="none" stroke="#5FA8FF" stroke-width="2"/;
const cnt=(s,re)=>(s.match(re)||[]).length;
const COL3={swing:'#5FA8FF',swing2:'#C9A2FF',swing3:'#FFB86B'};

/* ---------- ② 大波段 SENTI-1 ---------- */
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
    /* 等级标记：三角形数 == 窗口内信号数；光环数 == 窗口内 A 级数 */
    const [w0,w1]=winIdx(DATA[k]);
    const d0=DATA[k].dates[w0], d1=DATA[k].dates[w1];
    const inW=DATA[k].bottom.filter(e=>e.date>=d0&&e.date<=d1);
    const nTri=cnt(h,/l-6\.5 /g);
    const nHalo=cnt(h,/r="9\.5"/g);
    const nA=inW.filter(e=>e.grade==='A').length;
    if(nTri!==inW.length){console.log('  !! '+k+' '+rg+' 信号标记 '+nTri+' != 窗口信号 '+inW.length);bad2++;}
    if(nHalo!==nA){console.log('  !! '+k+' '+rg+' A级光环 '+nHalo+' != A级信号 '+nA);bad2++;}
    if(!info.includes(DATA[k].dates[DATA[k].dates.length-1])){
      console.log('  !! '+k+' '+rg+' 窗口信息缺少末日');bad2++;}
  }
  console.log('  '+k.padEnd(9)+line.join('  '));
}
/* 「20日平滑」已移除：图上必须永远是原始分值，与左侧卡片同一个值 */
RNG='all'; drawChart();
{
  const info=document.getElementById('rnginfo').innerHTML||'';
  const ok=/与左侧卡片同一个值/.test(info) && !/20日均/.test(info);
  console.log('  口径一致性: '+(ok?'✔ 图上为原始分值，与卡片同一个值':'✘ 图上口径与卡片不一致'));
  if(!ok) bad2++;
}
/* 卡片「最近信号」行（信号表已移除，这行是唯一的历史回溯入口） */
let badLs=0;
for(const k of Object.keys(DATA)){
  cur=k; render();
  const t=document.getElementById('lsig').innerHTML||'';
  if(!/级/.test(t)||/undefined|NaN/.test(t)){
    console.log('  !! '+k+' 最近信号行异常: '+t.slice(0,90));badLs++;}
}
if(badLs===0) console.log('  最近信号行: 6 个板块均已渲染 ✔');
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
  const ls=document.getElementById('lsig').innerHTML||'';
  const ms=document.getElementById('modestat').innerHTML||'';
  const lg=document.getElementById('legend').innerHTML||'';
  const gb=document.getElementById('gbody').innerHTML||'';
  if(/undefined|NaN/.test(ls+ms+lg+gb)){console.log('  !! '+k+' SWING 文案含 undefined/NaN');bad3++;}
  if(!/SWING|超卖/.test(lg)){console.log('  !! '+k+' 图例未切到 SWING');bad3++;}
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

/* ---------- ④ 信号源四选：合并（三张图）/ 回调底 / 动量拐点 / 均线拐点 ---------- */
console.log('\n④ 小波段信号源四选（合并 / 回调底 / 动量拐点 / 均线拐点）');
let bad4=0;
for(const s of ['union','swing','swing2','swing3']){
  SRC=s;
  let n=0, bad=0;
  for(const k of Object.keys(DATA)){
    cur=k; RNG='all'; drawChart(); render();
    const h=document.getElementById('plot').innerHTML||'';
    const ms=document.getElementById('modestat').innerHTML||'';
    const lg=document.getElementById('legend').innerHTML||'';
    const ls=document.getElementById('lsig').innerHTML||'';
    const info=document.getElementById('rnginfo').innerHTML||'';
    if(/undefined|NaN/.test(h+ms+lg+ls+info)){console.log('  !! '+s+' '+k+' 含 undefined/NaN');bad++;}
    if(s==='union'){
      /* 三张子图，各自一条主折线 + 一条阈值线 + 自己的信号点 */
      const nsvg=cnt(h,/<svg /g);
      if(nsvg!==3){console.log('  !! union '+k+' 子图数 '+nsvg+' != 3');bad++;}
      if(cnt(h,/class="subchart"/g)!==3){console.log('  !! union '+k+' 缺 subchart 容器');bad++;}
      if(cnt(h,/入场阈值/g)!==3){console.log('  !! union '+k+' 阈值线数 '+cnt(h,/入场阈值/g)+' != 3');bad++;}
      for(const key of ['swing','swing2','swing3']){
        const re=new RegExp('stroke="'+COL3[key]+'" stroke-width="1\\.8"');
        if(!re.test(h)){console.log('  !! union '+k+' 缺 '+key+' 主折线');bad++;}
      }
      if(!/三个模型各自一张图/.test(info)){console.log('  !! union '+k+' 窗口信息未说明三张图');bad++;}
      if(!/刻度不同/.test(lg)){console.log('  !! union '+k+' 图例未说明刻度不可比');bad++;}
      /* 信号点：src='both' 的事件三张图都画 → 期望三角形数 = Σ(1 or 3) */
      const [w0,w1]=winIdx(DATA[k]);
      const d0=DATA[k].dates[w0], d1=DATA[k].dates[w1];
      const inW=(DATA[k].union_events||[]).filter(e=>e.date>=d0&&e.date<=d1);
      const expTri=inW.reduce((a,e)=>a+((e.src==='both')?3:1),0);
      const nTri=cnt(h,/l-6 /g);
      if(nTri!==expTri){console.log('  !! union '+k+' 信号点 '+nTri+' != 期望 '+expTri);bad++;}
      /* 左侧卡片：三个最新值 + 隐藏单刻度条 */
      const sv=document.getElementById('sval');
      if(sv.className!=='big3'){console.log('  !! union '+k+' 卡片未切成三值布局');bad++;}
      if(!/回调底/.test(sv.innerHTML)||!/动量拐点/.test(sv.innerHTML)||!/均线拐点/.test(sv.innerHTML)){
        console.log('  !! union '+k+' 卡片缺某个模型的最新值');bad++;}
      if(document.getElementById('barscale').style.display!=='none'){
        console.log('  !! union '+k+' 单刻度条未隐藏');bad++;}
      n+=inW.length;
    }else{
      if(!MAIN_RE.test(h)){console.log('  !! '+s+' '+k+' 找不到主折线');bad++;}
      if(cnt(h,/<svg /g)!==1){console.log('  !! '+s+' '+k+' 单模型视图应有 1 张图');bad++;}
      n+=(DATA[k][s+'_events']||[]).length;
    }
  }
  console.log('  SRC='+s.padEnd(7)+' 6板块信号合计 '+String(n).padStart(4)
              +(bad?('  ✘ '+bad+' 处问题'):'  ✔'));
  bad4+=bad;
}
SRC='union';
/* 信号源名字的通俗解释必须存在（用户反馈「看不懂」才补的，别被后续改动删掉）。
   注意 stub 的 appendChild 是空函数、也没有 querySelector，所以直接读 srcHint.innerHTML。 */
const shTxt=((typeof srcHint!=='undefined' && srcHint.innerHTML)||'');
if(!/回调底/.test(shTxt)||!/动量拐点/.test(shTxt)||!/均线拐点/.test(shTxt)){
  console.log('  !! 信号源缺少通俗解释（需同时含「回调底」「动量拐点」「均线拐点」）');bad4++;}
console.log('\n  '+(bad4===0?'✔ 信号源四选冒烟通过':'✘ 信号源四选存在 '+bad4+' 处问题'));
process.exit((bad2+bad3+bad4)?1:0);
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
