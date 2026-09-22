"""页面冒烟测试：抓取线上页面 → 校验 JS 语法 → 用真实数据跑一遍表格渲染。

为什么要这个：模板里改 JS 很容易写出「语法没错但渲染出 undefined / 行数不对」的问题，
而浏览器起不来（Windows 上 agent-browser 不可用），肉眼也看不全 6 个板块 × 两种表。
这个脚本三步走：
  ① 抽出页面的 <script> 段，用 node --check 查语法
  ② 抽出页面里内联的 DATA / SUM（真实数据）
  ③ 在 node 里 stub 掉 document，真实调用 drawChart() 与渲染逻辑，校验点数/末点坐标/等级标记/NaN

用法：
  python _explore/smoke_page.py                # 默认 http://127.0.0.1:8779/
  python _explore/smoke_page.py --port 8780
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
for(const k of Object.keys(DATA)){
  cur=k;
  const line=[];
  for(const rg of ['7d','1m','3m','6m','1y','all']){
    RNG=rg; drawChart();
    const h=document.getElementById('plot').innerHTML||'';
    const info=document.getElementById('rnginfo').innerHTML||'';
    if(/undefined|NaN/.test(h+info)){console.log('  !! '+k+' '+rg+' 输出含 undefined/NaN');bad2++;continue;}
    const m=h.match(/<path d="([^"]+)" fill="none" stroke="#5FA8FF" stroke-width="2"/);
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
    const all=(MODE==='swing')?(DATA[k].swing_events||[]):DATA[k].bottom;
    const inW=all.filter(e=>e.date>=d0&&e.date<=d1);
    const nTri=(h.match(/l-6\.5 /g)||[]).length;
    const nHalo=(h.match(/r="9\.5"/g)||[]).length;
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

/* ---------- ③ 小波段 SWING 模式 ---------- */
console.log('\n③ 小波段 SWING 模式');
let bad3=0;
MODE='swing';
if(!(SUM.swing&&SUM.swing.total)){console.log('  !! SUM.swing 缺失');bad3++;}
for(const k of Object.keys(DATA)){
  cur=k;
  const line=[];
  for(const rg of ['7d','1m','3m','6m','1y','all']){
    RNG=rg; drawChart();
    const h=document.getElementById('plot').innerHTML||'';
    const info=document.getElementById('rnginfo').innerHTML||'';
    if(/undefined|NaN/.test(h+info)){console.log('  !! '+k+' '+rg+' 输出含 undefined/NaN');bad3++;continue;}
    const m=h.match(/<path d="([^"]+)" fill="none" stroke="#5FA8FF" stroke-width="2"/);
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
  const need=['2026-08-03','2026-09-14'];
  for(const d of need){
    const ok=ev.includes(d);
    console.log('  科创板 '+d+' 标记: '+(ok?'✔':'✘ 未触发'));
    if(!ok) bad3++;
  }
}
console.log('\n  '+(bad3===0?'✔ SWING 模式冒烟通过':'✘ SWING 存在 '+bad3+' 处问题'));
process.exit((bad2+bad3)?1:0);
"""


def fetch(url):
    req = urllib.request.Request(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=60).read().decode("utf-8", "replace")


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
