// Checks the class colours of the realtime viewer. Run by tests/test_viewer_colors.py:
//     node tests/viewer_colors_check.js pciSeq/src/realtime_viewer/static/js
//
// It loads the viewer's real state.js and colors.js into a bare sandbox (no browser)
// and checks the four rules the colours follow:
//     no colour scheme loaded           one colour per real class, however many there are
//     scheme loaded, class named in it  the user's colour
//     scheme loaded, class not named    grey
//     Zero, always                      black
// d3 is not around outside the browser, so the two d3 calls colors.js makes are stood
// in for here with the plain hsl to rgb sums.
const fs = require('fs'), vm = require('vm');
const J = process.argv[2];
// the real hsl -> rgb maths, standing in for d3.hsl(...).rgb()
const d3 = { hsl: (h, s, l) => ({ rgb: () => {
    const c = (1 - Math.abs(2 * l - 1)) * s, x = c * (1 - Math.abs((h / 60) % 2 - 1)), m = l - c / 2;
    const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x] : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
    return { r: (r + m) * 255, g: (g + m) * 255, b: (b + m) * 255 }; } }),
  rgb: (hex) => ({ r: parseInt(hex.slice(1, 3), 16), g: parseInt(hex.slice(3, 5), 16), b: parseInt(hex.slice(5, 7), 16) }) };
const sandbox = { window: {}, d3, console: { log() {}, warn: (m) => warnings.push(m) }, parseInt, Math, Object, Set,
                  document: { getElementById: () => ({ style: {}, textContent: '' }), addEventListener() {} }, setTimeout() {} };
const warnings = [];
vm.createContext(sandbox);
for (const f of ['state.js', 'colors.js']) vm.runInContext(fs.readFileSync(`${J}/${f}`, 'utf8'), sandbox);
const { state, colors } = sandbox.window.pciSeq;
const grey = '128,128,128', black = '0,0,0';
const setNames = (names) => { state.cellClassNames = {}; names.forEach((n, i) => state.cellClassNames[i] = n); };
const col = (i) => String(state.cellClassColors[i]);
let ok = true; const check = (what, cond) => { console.log((cond ? 'PASS  ' : 'FAIL  ') + what); ok = ok && cond; };

for (const nReal of [65, 300]) {
  const names = Array.from({ length: nReal }, (_, i) => `class_${String(i).padStart(3, '0')}`).concat(['Zero']);
  setNames(names); state.customColorScheme = null;
  colors.generateColorPalette(names);
  const real = names.slice(0, -1).map((_, i) => col(i));
  check(`${nReal + 1} classes, no scheme: every real class has a colour, none grey, none black`,
        real.every((c) => c !== 'undefined' && c !== grey && c !== black));
  check(`${nReal + 1} classes, no scheme: Zero (index ${nReal}) is black`, col(nReal) === black);
  check(`${nReal + 1} classes, no scheme: exactly ${nReal + 1} colours made`, Object.keys(state.cellClassColors).length === nReal + 1);

  colors.applyColorScheme({ class_000: '#ff0000', class_010: '#00ff00', Zero: '#ffffff' });
  check(`${nReal + 1} classes, scheme: a named class gets the user's colour`, col(0) === '255,0,0' && col(10) === '0,255,0');
  check(`${nReal + 1} classes, scheme: every class the scheme does not name is grey`,
        names.slice(0, -1).every((_, i) => i === 0 || i === 10 || col(i) === grey));
  check(`${nReal + 1} classes, scheme: Zero is black even though the scheme said white`, col(nReal) === black);

  // a new run in the same page: names arrive again, palette is rebuilt, the scheme must come back
  colors.generateColorPalette(names);
  colors.applyColorScheme(state.customColorScheme);
  check(`${nReal + 1} classes, new run: the remembered scheme is back on`, col(0) === '255,0,0' && col(5) === grey && col(nReal) === black);
}
state.cellClassNames = {}; colors.generateColorPalette();
check('before the class names arrive: the startup palette still has 65 colours', Object.keys(state.cellClassColors).length === 65);
console.log('warnings the viewer would print:', warnings.length ? warnings[warnings.length - 1].slice(0, 90) + '...' : 'none');
process.exit(ok ? 0 : 1);
