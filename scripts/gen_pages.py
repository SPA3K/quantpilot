#!/usr/bin/env python3
"""Generate self-contained static HTML for QuantPilot GitHub Pages."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__)) + "/.."

# Load data
with open(f"{BASE}/data/demo/results_compact.json") as f:
    demo_results = json.load(f)

with open(f"{BASE}/data/demo/strategies_compact.json") as f:
    presets = json.load(f)

results_js = json.dumps(demo_results, ensure_ascii=False, separators=(',', ':'))
presets_js = json.dumps(presets, ensure_ascii=False, separators=(',', ':'))

with open(f"{BASE}/eval_results/annual_backtest_20260820_1918.json") as f:
    ml_data = json.load(f)

ml_js = json.dumps(ml_data, ensure_ascii=False, separators=(',', ':'))

html = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantPilot — 量化策略交易平台</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚀</text></svg>">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0f1117;--sf:#1a1d27;--sf2:#242836;--bd:#2d3148;--tx:#e4e6ef;--dm:#8b8fa3;--ac:#6c5ce7;--a2:#a29bfe;--gn:#00cec9;--rd:#ff6b6b;--or:#fdcb6e;--r:12px}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,system-ui,sans-serif;line-height:1.6}
.app{max-width:1400px;margin:0 auto;padding:20px}
header{display:flex;align-items:center;justify-content:space-between;padding:16px 0 24px;border-bottom:1px solid var(--bd);margin-bottom:24px;flex-wrap:wrap;gap:12px}
.logo{font-size:24px;font-weight:700}.logo span{color:var(--ac)}
.tag{background:var(--ac);color:#fff;font-size:11px;padding:2px 8px;border-radius:20px;margin-left:8px}
.demo-tag{background:var(--or);color:#000;font-size:11px;padding:2px 8px;border-radius:4px;font-weight:600}
.builder-layout{display:grid;grid-template-columns:300px 1fr;gap:24px}
@media(max-width:900px){.builder-layout{grid-template-columns:1fr}}
.card{background:var(--sf);border:1px solid var(--bd);border-radius:var(--r);padding:20px}
.card h3{font-size:14px;color:var(--dm);text-transform:uppercase;letter-spacing:1px;margin-bottom:16px}
.catalog{position:sticky;top:20px;max-height:calc(100vh - 40px);overflow-y:auto}
.algo-card{background:var(--sf2);border:1px solid var(--bd);border-radius:8px;padding:12px;margin-bottom:8px;cursor:pointer;transition:all .2s}
.algo-card:hover{border-color:var(--ac);transform:translateX(4px)}
.algo-card.buy{border-left:3px solid var(--gn)}.algo-card.sell{border-left:3px solid var(--rd)}
.algo-card .name{font-weight:600;font-size:14px}.algo-card .desc{font-size:12px;color:var(--dm);margin-top:4px}
.tag-sm{font-size:10px;padding:1px 6px;border-radius:4px}
.tag-buy{background:rgba(0,206,201,.15);color:var(--gn)}.tag-sell{background:rgba(255,107,107,.15);color:var(--rd)}
.tabs{display:flex;gap:4px;margin-bottom:20px;background:var(--sf);border-radius:8px;padding:4px;border:1px solid var(--bd)}
.tab{flex:1;text-align:center;padding:10px;border-radius:6px;cursor:pointer;font-size:14px;font-weight:500;color:var(--dm);transition:all .2s}
.tab.active{background:var(--ac);color:#fff}.tab:hover:not(.active){color:var(--tx)}
.slot-group{margin-bottom:16px}.slot-label{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.slot-label.buy-l{color:var(--gn)}.slot-label.sell-l{color:var(--rd)}
.slot-area{min-height:60px;background:var(--sf2);border:2px dashed var(--bd);border-radius:8px;padding:12px;display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start}
.slot-area.drag-over{border-color:var(--ac);background:rgba(108,92,231,.05)}
.slot-item{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:10px 14px;position:relative;min-width:200px}
.slot-item.buy-slot{border-left:3px solid var(--gn)}.slot-item.sell-slot{border-left:3px solid var(--rd)}
.slot-item .slot-name{font-weight:600;font-size:13px}
.slot-item .remove-btn{position:absolute;top:6px;right:8px;background:none;border:none;color:var(--dm);cursor:pointer;font-size:16px}
.slot-item .remove-btn:hover{color:var(--rd)}
.param-row{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:12px}
.param-row label{min-width:70px;color:var(--dm)}
.param-row input[type=range]{flex:1;accent-color:var(--ac)}
.param-row .pv{min-width:50px;text-align:right;font-family:monospace;color:var(--a2)}
.config-row{display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap}
.config-field{display:flex;flex-direction:column;gap:4px}
.config-field label{font-size:12px;color:var(--dm)}
.config-field input,.config-field select{background:var(--sf2);border:1px solid var(--bd);border-radius:6px;padding:8px 12px;color:var(--tx);font-size:14px}
.config-field input:focus,.config-field select:focus{outline:none;border-color:var(--ac)}
.stock-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.stock-chip{background:var(--sf2);border:1px solid var(--bd);border-radius:20px;padding:4px 12px;font-size:12px;cursor:pointer;transition:all .2s}
.stock-chip.selected{background:var(--ac);border-color:var(--ac);color:#fff}
.stock-chip:hover:not(.selected){border-color:var(--a2)}
.run-btn{width:100%;padding:14px;background:linear-gradient(135deg,var(--ac),#8b5cf6);color:#fff;border:none;border-radius:8px;font-size:16px;font-weight:600;cursor:pointer;margin-top:16px;transition:all .2s}
.run-btn:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(108,92,231,.3)}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}
.metric-card{background:var(--sf2);border-radius:8px;padding:16px;text-align:center}
.metric-card .mv{font-size:28px;font-weight:700}.metric-card .ml{font-size:12px;color:var(--dm);margin-top:4px}
.metric-card.pos .mv{color:var(--gn)}.metric-card.neg .mv{color:var(--rd)}.metric-card.neu .mv{color:var(--a2)}
.chart-box{background:var(--sf2);border-radius:8px;padding:20px;margin-bottom:24px;position:relative;height:350px}
.preset-card{background:var(--sf2);border:1px solid var(--bd);border-radius:12px;padding:20px;margin-bottom:16px;cursor:pointer;transition:all .2s}
.preset-card:hover{border-color:var(--ac);transform:translateY(-2px);box-shadow:0 4px 16px rgba(108,92,231,.15)}
.preset-card .ptitle{font-size:18px;font-weight:600;margin-bottom:8px}
.preset-card .pdesc{font-size:13px;color:var(--dm);margin-bottom:12px}
.preset-metrics{display:flex;gap:16px;flex-wrap:wrap}
.preset-metrics span{font-size:13px}.preset-metrics .pos{color:var(--gn)}.preset-metrics .neg{color:var(--rd)}
.empty-state{text-align:center;padding:40px;color:var(--dm)}.empty-state .icon{font-size:48px;margin-bottom:12px}
</style>
</head>
<body>
<div class="app">
<header>
  <div class="logo">Quant<span>Pilot</span> <span class="tag">v0.3</span></div>
  <div style="display:flex;align-items:center;gap:8px">
    <span class="demo-tag">DEMO</span>
    <span style="color:var(--dm);font-size:13px">量化策略交易平台 · 12个免费策略积木</span>
  </div>
</header>

<div class="tabs">
  <div class="tab active" onclick="switchTab('build')">🔧 搭建策略</div>
  <div class="tab" onclick="switchTab('presets')">⭐ 策略预设</div>
  <div class="tab" onclick="switchTab('results')">📊 回测结果</div>
  <div class="tab" onclick="switchTab('catalog')">📚 组件手册</div>
  <div class="tab" onclick="switchTab('ml')">🧠 ML因子选股</div>
</div>

<div id="tab-build">
<div class="builder-layout">
  <div class="card catalog">
    <h3>策略积木库</h3>
    <input type="text" id="algo-search" placeholder="搜索组件..."
      style="width:100%;background:var(--sf2);border:1px solid var(--bd);border-radius:6px;padding:8px 12px;color:var(--tx);font-size:13px;margin-bottom:12px"
      oninput="renderAlgos(this.value)">
    <div id="algo-list"></div>
  </div>
  <div>
    <div class="card" style="margin-bottom:16px">
      <div class="slot-group">
        <div class="slot-label buy-l">📈 买入信号</div>
        <div class="slot-area" id="buy-slots"
          ondragover="event.preventDefault();this.classList.add('drag-over')"
          ondragleave="this.classList.remove('drag-over')"
          ondrop="dropAlgo(event,'buy')">
          <div class="empty-state" id="buy-empty" style="width:100%"><div class="icon">🎯</div>点击左侧组件添加</div>
        </div>
      </div>
      <div class="slot-group">
        <div class="slot-label sell-l">📉 卖出/止损</div>
        <div class="slot-area" id="sell-slots"
          ondragover="event.preventDefault();this.classList.add('drag-over')"
          ondragleave="this.classList.remove('drag-over')"
          ondrop="dropAlgo(event,'sell')">
          <div class="empty-state" id="sell-empty" style="width:100%"><div class="icon">🛡️</div>可选</div>
        </div>
      </div>
    </div>
    <div class="card">
      <h3>回测配置</h3>
      <div class="config-row">
        <div class="config-field" style="flex:1"><label>起始日期</label><input type="date" id="start-date" value="2023-01-01"></div>
        <div class="config-field" style="flex:1"><label>结束日期</label><input type="date" id="end-date" value="2025-12-31"></div>
        <div class="config-field" style="flex:1"><label>初始资金</label><input type="number" id="capital" value="100000" step="10000"></div>
      </div>
      <div style="margin-top:8px"><label style="font-size:12px;color:var(--dm)">选择股票</label>
        <div class="stock-chips" id="stock-chips"></div>
      </div>
      <button class="run-btn" onclick="runBacktest()">🚀 运行回测</button>
    </div>
  </div>
</div>
</div>

<div id="tab-presets" style="display:none"><div id="preset-list"></div></div>

<div id="tab-results" style="display:none">
  <div id="no-results" class="empty-state"><div class="icon">📊</div><div>还没有回测结果<br>去「搭建策略」或「策略预设」试试</div></div>
  <div id="results-content" style="display:none">
    <div style="margin-bottom:16px;display:flex;align-items:center;gap:8px">
      <h2 id="result-title" style="font-size:20px"></h2><span class="demo-tag">DEMO</span>
    </div>
    <div class="metrics-grid" id="metrics-grid"></div>
    <div class="chart-box"><canvas id="equity-chart"></canvas></div>
  </div>
</div>

<div id="tab-catalog" style="display:none">
  <div class="card"><h3>策略积木手册</h3><div id="catalog-detail"></div></div>
</div>

<div id="tab-ml" style="display:none">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:20px">
    <h2 style="font-size:20px">🧠 ML多层因子选股回测</h2><span class="demo-tag">DEMO</span>
  </div>
  <div class="card" style="margin-bottom:16px">
    <h3>因子模型概览</h3>
    <div style="overflow-x:auto"><table id="ml-factor-table" style="width:100%;border-collapse:collapse;font-size:14px"></table></div>
  </div>
  <div class="card" style="margin-bottom:16px">
    <h3>年度回测绩效</h3>
    <div style="overflow-x:auto"><table id="ml-year-table" style="width:100%;border-collapse:collapse;font-size:14px"></table></div>
  </div>
  <div class="card">
    <h3>个股选股详情</h3>
    <select id="ml-year-select" style="background:var(--sf2);border:1px solid var(--bd);border-radius:6px;padding:8px 12px;color:var(--tx);font-size:14px;margin-bottom:16px" onchange="renderMLDetail()"></select>
    <div id="ml-detail"></div>
  </div>
</div>
</div>

<script>
const ALGOS=[
{name:"双均线交叉",desc:"快线上穿慢线买入，下穿卖出",cat:"buy",params:[{n:"fast_period",d:5,mn:3,mx:60,l:"快线周期"},{n:"slow_period",d:20,mn:10,mx:250,l:"慢线周期"}]},
{name:"RSI超买超卖",desc:"RSI<30买入，RSI>70卖出",cat:"buy",params:[{n:"period",d:14,mn:6,mx:28,l:"计算周期"},{n:"oversold",d:30,mn:15,mx:40,l:"超卖阈值"},{n:"overbought",d:70,mn:60,mx:85,l:"超买阈值"}]},
{name:"MACD",desc:"DIF上穿DEA买入，下穿卖出",cat:"buy",params:[{n:"fast_period",d:12,mn:5,mx:26,l:"快线"},{n:"slow_period",d:26,mn:12,mx:52,l:"慢线"},{n:"signal_period",d:9,mn:5,mx:20,l:"信号线"}]},
{name:"布林带",desc:"触及下轨买入，触及上轨卖出",cat:"buy",params:[{n:"period",d:20,mn:10,mx:50,l:"周期"},{n:"num_std",d:2.0,mn:1.0,mx:3.0,l:"标准差"}]},
{name:"KDJ",desc:"K上穿D买入，K下穿D卖出",cat:"buy",params:[{n:"period",d:9,mn:5,mx:21,l:"周期"},{n:"k_smooth",d:3,mn:2,mx:7,l:"K平滑"},{n:"d_smooth",d:3,mn:2,mx:7,l:"D平滑"}]},
{name:"海龟交易法",desc:"突破N日高点买入，跌破M日低点卖出",cat:"buy",params:[{n:"entry_period",d:20,mn:10,mx:55,l:"入场"},{n:"exit_period",d:10,mn:5,mx:20,l:"出场"}]},
{name:"量价配合",desc:"放量上涨买入，缩量滞涨卖出",cat:"buy",params:[{n:"volume_ratio",d:1.5,mn:1.1,mx:3.0,l:"量比"},{n:"price_change",d:0.02,mn:0.005,mx:0.05,l:"涨幅"}]},
{name:"OBV能量潮",desc:"OBV趋势确认买入，顶背离卖出",cat:"buy",params:[{n:"obv_period",d:20,mn:5,mx:60,l:"OBV周期"}]},
{name:"网格交易",desc:"跌N%买入一格，涨N%卖出一格",cat:"buy",params:[{n:"grid_pct",d:3.0,mn:1.0,mx:10.0,l:"网格幅度%"},{n:"grid_levels",d:5,mn:2,mx:10,l:"网格层数"}]},
{name:"ATR追踪止损",desc:"跌破ATR追踪线时卖出",cat:"sell",params:[{n:"atr_period",d:14,mn:5,mx:30,l:"ATR周期"},{n:"atr_mult",d:2.0,mn:1.0,mx:4.0,l:"ATR倍数"}]},
{name:"止盈",desc:"收益达到目标时卖出",cat:"sell",params:[{n:"take_profit_pct",d:10,mn:3,mx:30,l:"目标收益%"}]},
{name:"止损",desc:"亏损超过阈值时卖出",cat:"sell",params:[{n:"stop_loss_pct",d:5,mn:2,mx:15,l:"止损阈值%"}]}
];
const STOCKS=["宁德时代","贵州茅台","比亚迪","中国平安","招商银行","隆基绿能","药明康德","长江电力"];
const DEMO_RESULTS=PLACEHOLDER_RESULTS;
const PRESETS=PLACEHOLDER_PRESETS;
const ML_DATA=PLACEHOLDER_ML;

let buySlots=[],sellSlots=[],chart=null;

function init(){renderAlgos();renderStocks();renderPresets();renderCatalog();renderML()}

function renderAlgos(f=''){
  const el=document.getElementById('algo-list');const q=f.toLowerCase();
  el.innerHTML=ALGOS.filter(a=>!q||a.name.includes(q)||a.desc.includes(q)).map(a=>
    '<div class="algo-card '+a.cat+'" draggable="true" ondragstart="event.dataTransfer.setData(\'text/plain\',\''+a.name+'\')" onclick="clickAlgo(\''+a.name+'\')">'+
    '<div style="display:flex;justify-content:space-between;align-items:center"><span class="name">'+a.name+'</span><span class="tag-sm tag-'+a.cat+'">'+(a.cat==='buy'?'买入':'卖出')+'</span></div>'+
    '<div class="desc">'+a.desc+'</div>'+
    '<div style="font-size:11px;color:var(--dm);margin-top:6px">'+a.params.map(p=>p.n+'='+p.d).join(', ')+'</div>'+
    '</div>').join('');
}

function clickAlgo(n){var a=ALGOS.find(x=>x.name===n);if(!a)return;addSlot(n,a.cat==='sell'?'sell':'buy')}
function dropAlgo(e,t){e.preventDefault();e.currentTarget.classList.remove('drag-over');addSlot(e.dataTransfer.getData('text/plain'),t)}
function addSlot(n,t){
  var a=ALGOS.find(x=>x.name===n);if(!a)return;var p={};
  a.params.forEach(function(x){p[x.n]=x.d});
  if(t==='buy'){buySlots.push({algo:n,params:p});document.getElementById('buy-empty').style.display='none'}
  else{sellSlots.push({algo:n,params:p});document.getElementById('sell-empty').style.display='none'}
  renderSlots(t);
}
function removeSlot(t,i){
  if(t==='buy'){buySlots.splice(i,1);if(!buySlots.length)document.getElementById('buy-empty').style.display=''}
  else{sellSlots.splice(i,1);if(!sellSlots.length)document.getElementById('sell-empty').style.display=''}
  renderSlots(t);
}
function renderSlots(t){
  var c=document.getElementById(t+'-slots');
  c.querySelectorAll('.slot-item').forEach(function(x){x.remove()});
  var slots=t==='buy'?buySlots:sellSlots;
  slots.forEach(function(s,i){
    var a=ALGOS.find(function(x){return x.name===s.algo});
    var d=document.createElement('div');d.className='slot-item '+t+'-slot';
    d.innerHTML='<button class="remove-btn" onclick="removeSlot(\''+t+'\','+i+')">✕</button>'+
      '<div class="slot-name">'+s.algo+'</div>'+
      a.params.map(function(p){
        return '<div class="param-row"><label>'+p.l+'</label>'+
          '<input type="range" min="'+p.mn+'" max="'+p.mx+'" step="'+(Number.isInteger(p.d)?1:0.01)+'" value="'+s.params[p.n]+'" oninput="updParam(\''+t+'\','+i+',\''+p.n+'\',this.value)">'+
          '<span class="pv" id="pv-'+t+'-'+i+'-'+p.n+'">'+s.params[p.n]+'</span></div>';
      }).join('');
    c.appendChild(d);
  });
}
function updParam(t,i,n,v){
  var s=(t==='buy'?buySlots:sellSlots)[i];
  var a=ALGOS.find(function(x){return x.name===s.algo});
  var pd=a.params.find(function(x){return x.n===n});
  s.params[n]=pd.d%1===0?parseInt(v):parseFloat(v);
  document.getElementById('pv-'+t+'-'+i+'-'+n).textContent=s.params[n];
}
function renderStocks(){
  document.getElementById('stock-chips').innerHTML=STOCKS.map(function(s){
    return '<div class="stock-chip" onclick="this.classList.toggle(\'selected\')">'+s+'</div>';
  }).join('');
}

function runBacktest(){
  var sel=Array.from(document.querySelectorAll('.stock-chip.selected')).map(function(e){return e.textContent});
  if(!sel.length){alert('请至少选择一只股票');return}
  if(!buySlots.length){alert('请至少添加一个买入信号');return}
  var stock=sel[0],algo=buySlots[0]?buySlots[0].algo:'双均线交叉';
  var keys=Object.keys(DEMO_RESULTS);
  var key=keys.find(function(k){return k.indexOf(stock)>=0&&k.indexOf(algo)>=0})
    ||keys.find(function(k){return k.indexOf(stock)>=0})
    ||keys[0];
  var demo=DEMO_RESULTS[key];
  if(!demo){alert('该组合无演示数据');return}
  showResult(demo,demo.stock+' × '+demo.algorithm);
}

function showResult(d,title){
  document.getElementById('no-results').style.display='none';
  document.getElementById('results-content').style.display='';
  document.getElementById('result-title').textContent=title;
  var m=d.metrics;
  var items=[
    {l:'总收益',v:m.total_return+'%',c:m.total_return>=0?'pos':'neg'},
    {l:'年化收益',v:m.annual_return+'%',c:m.annual_return>=0?'pos':'neg'},
    {l:'最大回撤',v:m.max_drawdown+'%',c:'neg'},
    {l:'夏普比',v:m.sharpe_ratio,c:m.sharpe_ratio>=1?'pos':m.sharpe_ratio>=0?'neu':'neg'},
    {l:'胜率',v:m.win_rate+'%',c:m.win_rate>=50?'pos':'neu'},
    {l:'交易次数',v:m.total_trades,c:'neu'}
  ];
  document.getElementById('metrics-grid').innerHTML=items.map(function(i){
    return '<div class="metric-card '+i.c+'"><div class="mv">'+i.v+'</div><div class="ml">'+i.l+'</div></div>';
  }).join('');
  if(chart)chart.destroy();
  var eq=d.equity_curve;
  chart=new Chart(document.getElementById('equity-chart').getContext('2d'),{
    type:'line',
    data:{labels:eq.map(function(p){return p.date}),datasets:[{label:'净值',data:eq.map(function(p){return p.value}),borderColor:'#6c5ce7',backgroundColor:'rgba(108,92,231,.1)',fill:true,tension:.3,pointRadius:0,borderWidth:2}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:{ticks:{maxTicksLimit:10,color:'#8b8fa3',font:{size:11}},grid:{color:'#2d3148'}},
        y:{ticks:{color:'#8b8fa3',callback:function(v){return '¥'+v.toLocaleString()}},grid:{color:'#2d3148'}}}}
  });
  switchTab('results');
}

function renderPresets(){
  document.getElementById('preset-list').innerHTML=Object.entries(PRESETS).map(function(kv){
    var k=kv[0],s=kv[1],m=s.metrics,ret=m.total_return;
    return '<div class="preset-card" onclick="showPresetResult(\''+k+'\')">'+
      '<div class="ptitle">'+s.name+'</div>'+
      '<div class="pdesc">股票: '+s.stocks.join('、')+'</div>'+
      '<div class="preset-metrics">'+
      '<span>总收益: <span class="'+(ret>=0?'pos':'neg')+'">'+ret+'%</span></span>'+
      '<span>夏普比: <span class="'+(m.sharpe_ratio>=1?'pos':'neg')+'">'+m.sharpe_ratio+'</span></span>'+
      '<span>胜率: '+m.win_rate+'%</span>'+
      '<span>交易: '+m.total_trades+'次</span>'+
      '</div></div>';
  }).join('');
}
function showPresetResult(k){
  var s=PRESETS[k];if(!s)return;
  showResult({stock:s.stocks.join(','),algorithm:s.name,metrics:s.metrics,equity_curve:s.equity_curve},'⭐ '+s.name);
}

function renderCatalog(){
  document.getElementById('catalog-detail').innerHTML=ALGOS.map(function(a){
    return '<div style="margin-bottom:20px;padding:16px;background:var(--sf2);border-radius:8px;border-left:3px solid '+(a.cat==='buy'?'var(--gn)':'var(--rd)')+'">'+
      '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'+
      '<div style="font-size:18px;font-weight:600">'+a.name+'</div><span class="tag-sm tag-'+a.cat+'">'+(a.cat==='buy'?'买入':'卖出')+'</span></div>'+
      '<div style="color:var(--dm);margin-bottom:12px">'+a.desc+'</div>'+
      a.params.map(function(p){
        return '<div style="font-size:13px;margin-bottom:4px"><span style="color:var(--a2);font-family:monospace;min-width:100px;display:inline-block">'+p.n+'</span>'+
          '<span style="color:var(--dm)">默认'+p.d+' · '+p.mn+'~'+p.mx+'</span><span style="margin-left:8px">'+p.l+'</span></div>';
      }).join('')+
      '<button onclick="clickAlgo(\''+a.name+'\')" style="margin-top:12px;background:var(--ac);color:#fff;border:none;border-radius:6px;padding:6px 16px;cursor:pointer;font-size:13px">+ 添加到策略</button>'+
      '</div>';
  }).join('');
}

function renderML(){
  if(!ML_DATA||!ML_DATA.length)return;
  var ft='<thead><tr style="border-bottom:2px solid var(--bd)">';
  ft+='<th style="text-align:left;padding:10px 12px">因子模型</th>';
  ft+='<th style="text-align:left;padding:10px 12px">层级</th>';
  ft+='<th style="text-align:right;padding:10px 12px">融合权重</th>';
  ft+='<th style="text-align:right;padding:10px 12px">平均IC</th>';
  ft+='<th style="text-align:left;padding:10px 12px">IC评级</th></tr></thead><tbody>';
  var factors=[
    {name:'AlphaForge',lv:'L1',wt:'70%',ic:0.27},
    {name:'TechPulse',lv:'L0',wt:'20%',ic:0.04},
    {name:'Sentinel',lv:'L3',wt:'10%',ic:-0.10},
    {name:'融合模型',lv:'Fusion',wt:'100%',ic:0.17}
  ];
  factors.forEach(function(f){
    var cls=f.ic>=0.2?'pos':f.ic>=0?'neu':'neg';
    var bar=Math.min(Math.abs(f.ic)/0.3*100,100);
    ft+='<tr style="border-bottom:1px solid var(--bd)">';
    ft+='<td style="padding:10px 12px;font-weight:600">'+f.name+'</td>';
    ft+='<td style="padding:10px 12px;color:var(--a2);font-family:monospace">'+f.lv+'</td>';
    ft+='<td style="padding:10px 12px;text-align:right">'+f.wt+'</td>';
    ft+='<td style="padding:10px 12px;text-align:right;font-weight:600;color:'+(f.ic>=0?'var(--gn)':'var(--rd)')+'">'+(f.ic>=0?'+':'')+f.ic.toFixed(2)+'</td>';
    ft+='<td style="padding:10px 12px"><div style="background:var(--sf);border-radius:4px;height:16px;width:120px;display:inline-block;vertical-align:middle"><div style="background:'+(f.ic>=0?'var(--gn)':'var(--rd)')+';height:100%;border-radius:4px;width:'+bar+'%"></div></div></td>';
    ft+='</tr>';
  });
  ft+='</tbody>';
  document.getElementById('ml-factor-table').innerHTML=ft;

  var avgLS=0;ML_DATA.forEach(function(d){avgLS+=d.long_short});avgLS/=ML_DATA.length;
  var yt='<thead><tr style="border-bottom:2px solid var(--bd)">';
  yt+='<th style="text-align:left;padding:10px 12px">年份</th>';
  ['TOP5收益','BOTTOM5收益','Long-Short','全市场','AlphaForge IC','TechPulse IC','Sentinel IC','Fusion IC'].forEach(function(h){
    yt+='<th style="text-align:right;padding:10px 12px">'+h+'</th>';
  });
  yt+='</tr></thead><tbody>';
  ML_DATA.forEach(function(d){
    var l0Sum=0,l1Sum=0,l3Sum=0,fuSum=0,n=0;
    Object.values(d.predictions).forEach(function(p){l0Sum+=p.l0;l1Sum+=p.l1;l3Sum+=p.l3;fuSum+=p.fusion;n++});
    var l0Avg=l0Sum/n,l1Avg=l1Sum/n,l3Avg=l3Sum/n,fuAvg=fuSum/n;
    yt+='<tr style="border-bottom:1px solid var(--bd)">';
    yt+='<td style="padding:10px 12px;font-weight:600">'+d.year+'</td>';
    yt+='<td style="padding:10px 12px;text-align:right;color:var(--gn)">'+(d.top5_ret>=0?'+':'')+d.top5_ret+'%</td>';
    yt+='<td style="padding:10px 12px;text-align:right;color:'+(d.bottom5_ret>=0?'var(--gn)':'var(--rd)')+'">'+(d.bottom5_ret>=0?'+':'')+d.bottom5_ret+'%</td>';
    yt+='<td style="padding:10px 12px;text-align:right;font-weight:700;color:'+(d.long_short>=0?'var(--gn)':'var(--rd)')+'">'+(d.long_short>=0?'+':'')+d.long_short+'%</td>';
    yt+='<td style="padding:10px 12px;text-align:right;color:'+(d.all_ret>=0?'var(--gn)':'var(--rd)')+'">'+(d.all_ret>=0?'+':'')+d.all_ret+'%</td>';
    [[l1Avg,'AlphaForge'],[l0Avg,'TechPulse'],[l3Avg,'Sentinel'],[fuAvg,'Fusion']].forEach(function(pair){
      var v=pair[0];
      yt+='<td style="padding:10px 12px;text-align:right;font-family:monospace;color:'+(v>=0?'var(--gn)':'var(--rd)')+'">'+(v>=0?'+':'')+v.toFixed(3)+'</td>';
    });
    yt+='</tr>';
  });
  yt+='<tr style="border-top:2px solid var(--bd);font-weight:700">';
  yt+='<td style="padding:10px 12px">平均</td>';
  yt+='<td style="padding:10px 12px;text-align:right">—</td><td style="padding:10px 12px;text-align:right">—</td>';
  yt+='<td style="padding:10px 12px;text-align:right;color:var(--gn)">+'+avgLS.toFixed(2)+'%</td>';
  yt+='<td style="padding:10px 12px;text-align:right">—</td>';
  yt+='<td style="padding:10px 12px;text-align:right;color:var(--gn)">+0.27</td>';
  yt+='<td style="padding:10px 12px;text-align:right;color:var(--gn)">+0.04</td>';
  yt+='<td style="padding:10px 12px;text-align:right;color:var(--rd)">-0.10</td>';
  yt+='<td style="padding:10px 12px;text-align:right;color:var(--gn)">+0.17</td>';
  yt+='</tr></tbody>';
  document.getElementById('ml-year-table').innerHTML=yt;

  var sel=document.getElementById('ml-year-select');
  ML_DATA.forEach(function(d,i){
    var o=document.createElement('option');o.value=i;o.textContent=d.year+'年';sel.appendChild(o);
  });
  renderMLDetail();
}
function renderMLDetail(){
  var idx=parseInt(document.getElementById('ml-year-select').value)||0;
  var d=ML_DATA[idx];if(!d)return;
  var preds=Object.entries(d.predictions).map(function(kv){return{code:kv[0],name:kv[1].name,l0:kv[1].l0,l1:kv[1].l1,l3:kv[1].l3,fusion:kv[1].fusion,ret:kv[1].actual_return}});
  var sorted=preds.slice().sort(function(a,b){return b.fusion-a.fusion});
  var top5=sorted.slice(0,5),bot5=sorted.slice(-5).reverse();

  function mkTable(title,stocks,cls){
    var h='<div style="margin-bottom:20px">';
    h+='<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">';
    h+='<span style="font-size:15px;font-weight:600">'+title+'</span>';
    h+='<span style="font-size:12px;padding:2px 8px;border-radius:20px;background:'+(cls==='top'?'rgba(0,206,201,.15)':'rgba(255,107,107,.15)')+';color:'+(cls==='top'?'var(--gn)':'var(--rd)')+'">'+stocks.length+'只</span></div>';
    h+='<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:13px">';
    h+='<thead><tr style="border-bottom:2px solid var(--bd)">';
    ['排名','代码','名称','TechPulse(L0)','AlphaForge(L1)','Sentinel(L3)','融合评分','实际收益'].forEach(function(t){
      h+='<th style="text-align:'+(t==='排名'||t==='代码'||t==='名称'?'left':'right')+';padding:8px 10px;color:var(--dm)">'+t+'</th>';
    });
    h+='</tr></thead><tbody>';
    stocks.forEach(function(s,i){
      h+='<tr style="border-bottom:1px solid var(--bd)">';
      h+='<td style="padding:8px 10px;color:var(--dm)">'+(i+1)+'</td>';
      h+='<td style="padding:8px 10px;font-family:monospace">'+s.code+'</td>';
      h+='<td style="padding:8px 10px;font-weight:600">'+s.name+'</td>';
      h+='<td style="padding:8px 10px;text-align:right;font-family:monospace;color:'+(s.l0>=0?'var(--gn)':'var(--rd)')+'">'+(s.l0>=0?'+':'')+s.l0.toFixed(3)+'</td>';
      h+='<td style="padding:8px 10px;text-align:right;font-family:monospace;color:'+(s.l1>=0?'var(--gn)':'var(--rd)')+'">'+(s.l1>=0?'+':'')+s.l1.toFixed(3)+'</td>';
      h+='<td style="padding:8px 10px;text-align:right;font-family:monospace;color:'+(s.l3>=0?'var(--gn)':'var(--rd)')+'">'+(s.l3>=0?'+':'')+s.l3.toFixed(3)+'</td>';
      h+='<td style="padding:8px 10px;text-align:right;font-weight:700;font-family:monospace;color:var(--a2)">'+s.fusion.toFixed(3)+'</td>';
      if(s.ret!==null){h+='<td style="padding:8px 10px;text-align:right;font-weight:600;color:'+(s.ret>=0?'var(--gn)':'var(--rd)')+'">'+(s.ret>=0?'+':'')+s.ret+'%</td>'}
      else{h+='<td style="padding:8px 10px;text-align:right;color:var(--dm)">N/A</td>'}
      h+='</tr>';
    });
    h+='</tbody></table></div></div>';
    return h;
  }
  document.getElementById('ml-detail').innerHTML=mkTable('🔥 TOP5 高分选股',top5,'top')+mkTable('❄️ BOTTOM5 低分选股',bot5,'bot');
}

function switchTab(n){
  document.querySelectorAll('.tab').forEach(function(t,i){
    t.classList.toggle('active',['build','presets','results','catalog','ml'][i]===n);
  });
  ['build','presets','results','catalog','ml'].forEach(function(id){
    document.getElementById('tab-'+id).style.display=id===n?'':'none';
  });
}

init();
</script>
</body>
</html>'''

# Inject the data
html = html.replace('PLACEHOLDER_RESULTS', results_js)
html = html.replace('PLACEHOLDER_PRESETS', presets_js)
html = html.replace('PLACEHOLDER_ML', ml_js)

out_path = f"{BASE}/docs/index.html"
with open(out_path, 'w') as f:
    f.write(html)

print(f"✅ Written {len(html):,} bytes ({len(html)/1024:.1f}KB) to {out_path}")
