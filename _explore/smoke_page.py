"""页面冒烟测试：抓取线上页面 → 校验 JS 语法 → 用真实数据跑一遍表格渲染。

为什么要这个：模板里改 JS 很容易写出「语法没错但渲染出 undefined / 行数不对」的问题，
而浏览器起不来（Windows 上 agent-browser 不可用），肉眼也看不全 6 个板块 × 两种表。
这个脚本三步走：
  ① 抽出页面的 <script> 段，用 node --check 查语法
  ② 抽出页面里内联的 DATA / SUM（真实数据）
  ③ 在 node 里 stub 掉 document，真实调用 table()，比对渲染行数与数据条数，
     并检查输出里有没有 undefined / NaN

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
    const m=h.match(/<path d="([^"]+)" fill="none" stroke="#58a6ff" stroke-width="2"/);
    if(!m){console.log('  !! '+k+' '+rg+' 找不到主折线');bad2++;continue;}
    const pts=(m[1].match(/L/g)||[]).length+1;
    const exp=expectN(DATA[k].dates,rg);
    const lastX=parseFloat(m[1].slice(m[1].lastIndexOf('L')+1).trim().split(' ')[0]);
    const okP=(pts===exp), okX=(Math.abs(lastX-782)<0.6);   // 末点应贴住绘图区右边界 62+720
    line.push(rg+':'+pts+(okP?'':'(应'+exp+')')+(okX?'':'(末点x='+lastX+')'));
    if(!okP||!okX) bad2++;
    if(!info.includes(DATA[k].dates[DATA[k].dates.length-1])){
      console.log('  !! '+k+' '+rg+' 窗口信息缺少末日');bad2++;}
  }
  console.log('  '+k.padEnd(9)+line.join('  '));
}
/* 平滑模式：图上是 20 日均，必须显式说明，不能与卡片静默不一致 */
RNG='all'; document.getElementById('smd').checked=true; drawChart();
{
  const info=document.getElementById('rnginfo').innerHTML||'';
  const ok=/20日均/.test(info);
  console.log('  平滑模式提示: '+(ok?'✔ 已说明「图上为 20日均」':'✘ 未说明'));
  if(!ok) bad2++;
}
console.log('\n  '+(bad2===0?'✔ 图表冒烟通过':'✘ 图表存在 '+bad2+' 处问题'));
process.exit(bad2?1:0);
"""


def fetch(url):
    req = urllib.request.Request(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=60).read().decode("utf-8", "replace")


JS_TEST = r"""
const fs=require('fs');
const {DATA,SUM}=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
global.SUM=SUM; global.HOLD=20;
global.GC={A:['A 高置信','#7ee787'],B:['B 中','#e3b341'],C:['C 低','#8b949e']};
global.fmt=(v,d=2)=>Number(v).toFixed(d);
global.document={getElementById:id=>({checked:false})};
const src=fs.readFileSync(process.argv[3],'utf8');
const a=src.indexOf('function gbadge'), b=src.indexOf('/* ---------- render ---------- */');
if(a<0||b<0){console.log('!! 未定位到函数段（模板结构可能变了）');process.exit(1);}
eval(src.slice(a,b));
let bad=0, tot=0;
for(const k of Object.keys(DATA)){
  const v=DATA[k];
  const tb=table(v.bottom,'bottom'), tt=table(v.top,'top');
  const nb=(tb.match(/<tr><td>/g)||[]).length, nt=(tt.match(/<tr><td>/g)||[]).length;
  tot+=nb+nt;
  const ok=nb===v.bottom.length && nt===v.top.length;
  if(!ok) bad++;
  console.log('  '+k.padEnd(9)+' 底部 '+nb+'/'+v.bottom.length+'   顶部 '+nt+'/'+v.top.length+(ok?'':'   <<< 行数不一致'));
  if(/undefined|NaN/.test(tb+tt)){console.log('     !! 输出含 undefined/NaN'); bad++;}
}
global.document={getElementById:id=>({checked:true})};
const fa=(table(DATA.CSI1000.bottom,'bottom').match(/<tr><td>/g)||[]).length;
const ea=DATA.CSI1000.bottom.filter(e=>e.grade==='A').length;
console.log('  A级筛选: 渲染 '+fa+' 行 / 数据 '+ea+' 个'+(fa===ea?'  ✔':'  <<< 不一致'));
if(fa!==ea) bad++;
console.log('\n  共渲染 '+tot+' 行；'+(bad===0?'✔ 冒烟测试通过':'✘ 存在 '+bad+' 处问题'));
process.exit(bad?1:0);
"""


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

    print("\n③ 用真实数据渲染表格")
    t_path = os.path.join(TMP, "test.js")
    with open(t_path, "w", encoding="utf-8") as f:
        f.write(JS_TEST)
    r = subprocess.run([NODE, t_path, d_path, js_path], capture_output=True, text=True)
    print((r.stdout or "").rstrip())
    if r.returncode != 0:
        print((r.stderr or "")[:1500]); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
