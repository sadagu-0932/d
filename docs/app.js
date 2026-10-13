const out = document.getElementById('out');
const input = document.getElementById('page');
let index = null;

function tex(s, display) {
  const el = document.createElement(display ? 'div' : 'span');
  try { katex.render(s, el, { displayMode: display, throwOnError: false }); }
  catch { el.textContent = s; }
  return el;
}

function msg(text) {
  out.innerHTML = '';
  const p = document.createElement('p');
  p.className = 'msg';
  p.textContent = text;
  out.appendChild(p);
}

async function loadIndex() {
  if (!index) index = await (await fetch('data/index.json')).json();
  return index;
}

async function show(page) {
  const idx = await loadIndex();
  input.value = page;
  history.replaceState(null, '', '#p=' + page);
  const info = idx.pages[page];
  if (!info) {
    msg(`${page}쪽 풀이는 아직 없습니다. (현재 ${idx.min}~${idx.max}쪽 준비됨)`);
    return;
  }
  if (!info.problems) {
    msg(`${page}쪽은 개념 정리 쪽이라 풀 문제가 없습니다.`);
    return;
  }
  const data = await (await fetch(`data/p${page}.json`)).json();
  out.innerHTML = '';
  if (data.section) {
    const h = document.createElement('p');
    h.className = 'sec';
    h.textContent = `${page}쪽 · ${data.section}`;
    out.appendChild(h);
  }
  for (const pr of data.problems) {
    const card = document.createElement('section');
    card.className = 'card';
    const h2 = document.createElement('h2');
    const no = document.createElement('span');
    no.className = 'no';
    no.textContent = pr.no;
    const ans = document.createElement('span');
    ans.className = 'ans';
    ans.append('답 ', tex(pr.answer, false));
    ans.addEventListener('click', () => ans.classList.toggle('shown'));
    h2.append(no, ans);
    const ol = document.createElement('ul');
    ol.className = 'steps';
    for (const st of pr.steps) {
      const li = document.createElement('li');
      if (st.text !== undefined) { li.className = 't'; li.textContent = st.text; }
      else li.appendChild(tex(st.tex, true));
      ol.appendChild(li);
    }
    card.append(h2, ol);
    out.appendChild(card);
  }
}

function step(d) {
  const p = parseInt(input.value, 10);
  if (p) show(p + d);
}

document.getElementById('go').addEventListener('submit', e => {
  e.preventDefault();
  const p = parseInt(input.value, 10);
  if (p) show(p);
});
document.getElementById('prev').addEventListener('click', () => step(-1));
document.getElementById('next').addEventListener('click', () => step(1));
const hide = document.getElementById('hideAns');
hide.addEventListener('change', () => document.body.classList.toggle('hiding', hide.checked));

const m = location.hash.match(/p=(\d+)/);
if (m) show(parseInt(m[1], 10));
