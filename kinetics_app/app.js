// === Configuration and State ===
let state = {
    trueParams: { n: 0, k: 0, CA0: 0, T: 0 },
    data: [], // Array of {t, CA}
    diffChartItem: null,
    intChartItem: null,
    currentIntOrder: 0
};

// Colors for dark mode chart
Chart.defaults.color = '#e0e0e0';

// === Mathematical Utilities ===

// Random number between min and max
function uniform(min, max) {
    return Math.random() * (max - min) + min;
}

// Generate normally distributed random variable using Box-Muller transform
function randomNormal(mean, stdDev) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0 * stdDev + mean;
}

// Simple Ordinary Least Squares (OLS) Linear Regression: y = m*x + b
function linearRegression(x, y) {
    const n = x.length;
    let sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0, sum_yy = 0;
    for (let i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += (x[i] * y[i]);
        sum_xx += (x[i] * x[i]);
        sum_yy += (y[i] * y[i]);
    }
    const denominator = (n * sum_xx - sum_x * sum_x);
    if(denominator === 0) return { slope: 0, intercept: 0, r2: 0 };
    
    const slope = (n * sum_xy - sum_x * sum_y) / denominator;
    const intercept = (sum_y - slope * sum_x) / n;
    
    // Calculate R-squared
    const yMean = sum_y / n;
    let ssTot = 0, ssRes = 0;
    for (let i = 0; i < n; i++) {
        const yPred = slope * x[i] + intercept;
        ssTot += Math.pow(y[i] - yMean, 2);
        ssRes += Math.pow(y[i] - yPred, 2);
    }
    const r2 = 1 - (ssRes / ssTot);
    return { slope, intercept, r2 };
}

// 4th Order Runge-Kutta solver for dC/dt = -k * C^n
function rk4Solve(CA0, k, n, t_end, num_points) {
    const data = [];
    const dt = t_end / num_points;
    let CA = CA0;
    let t = 0;
    
    data.push({ t: parseFloat(t.toFixed(3)), CA: parseFloat(CA.toFixed(4)) });
    
    // Safety check for CA to stay positive, especially for fractional or high orders
    for (let i = 1; i <= num_points; i++) {
        if (CA <= 0) {
            CA = 0;
            data.push({ t: parseFloat((t + dt).toFixed(3)), CA: 0 });
            t += dt;
            continue;
        }
        
        const f = (c) => -k * Math.pow(c, n);
        
        const k1 = f(CA);
        let k2 = f(Math.max(CA + 0.5 * dt * k1, 0));
        let k3 = f(Math.max(CA + 0.5 * dt * k2, 0));
        let k4 = f(Math.max(CA + dt * k3, 0));
        
        CA = CA + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
        if(CA < 0.0001) CA = 0; // Cap at 0
        t += dt;
        data.push({ t: parseFloat(t.toFixed(3)), CA: parseFloat(CA.toFixed(4)) });
    }
    
    return data;
}

function getKUnits(n) {
    if (n === 0) return "[mol / L*min]";
    else if (n === 0.5) return "[mol<sup>0.5</sup> / (L<sup>0.5</sup> &middot; min)]";
    else if (n === 1) return "[min<sup>-1</sup>]";
    else if (n === 1.5) return "[L<sup>0.5</sup> / (mol<sup>0.5</sup> &middot; min)]";
    else if (n === 2) return "[L / (mol &middot; min)]";
    else if (n === 3) return "[L<sup>2</sup> / (mol<sup>2</sup> &middot; min)]";
    else return `[(L / mol)<sup>${n-1}</sup> &middot; min<sup>-1</sup>]`;
}

// === Application Logic ===

function generateProblem() {
    // 1. Generate random parameters
    const orders = [0, 1, 2, 3, 0.5, 1.5]; // Common educational reaction orders
    const n = orders[Math.floor(Math.random() * orders.length)];
    const k = uniform(0.01, 0.2);
    const CA0 = uniform(0.5, 3.0);
    const T = uniform(298, 350); // Temperature just for context
    
    // Determine reasonable t_end based on half-life roughly to get good curves
    let t_end = 50; 
    if (n === 1) t_end = (Math.log(2) / k) * 3;
    else if (n === 2) t_end = (1 / (k * CA0)) * 3;
    else if (n === 3) t_end = (3 / (2 * k * CA0 * CA0)) * 2;
    else if (n === 0) t_end = (CA0 / k) * 0.9;
    
    if (t_end > 200) t_end = 150; // Cap maximum time
    
    state.trueParams = { n, k, CA0, Math: Math.round(T) };
    
    const num_points = Math.floor(uniform(8, 15));
    
    // 2. Solve exactly using RK4 (or analytical)
    let rawData = rk4Solve(CA0, k, n, t_end, num_points);
    
    // 3. Add experimental noise relative to reading value (heteroscedastic)
    state.data = rawData.map(pt => {
        // Noise is ~2% of the value plus a small flat base noise
        const noise = randomNormal(0, (0.02 * pt.CA + 0.005));
        let noisyCA = pt.CA + noise;
        if(noisyCA < 0) noisyCA = 0;
        return { t: pt.t, CA: parseFloat(noisyCA.toFixed(3)) };
    });
    
    // Render the new problem
    renderTable();
    performDifferentialAnalysis();
    performIntegralAnalysis();
    
    // Reset Reveal box
    document.getElementById('params-text').innerHTML = `Parámetros ocultos... ¿Puedes deducirlos?`;
    
    // Reset view state: hide results, show resolve button
    document.getElementById('diff-panel').classList.add('hidden');
    document.getElementById('int-panel').classList.add('hidden');
    document.getElementById('btn-resolve').classList.remove('hidden');
}

function renderTable() {
    const tbody = document.querySelector('#data-table tbody');
    tbody.innerHTML = '';
    state.data.forEach(pt => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${pt.t.toFixed(1)}</td><td>${pt.CA.toFixed(3)}</td>`;
        tbody.appendChild(tr);
    });
}

function performDifferentialAnalysis() {
    const { data } = state;
    if (data.length < 3) return;
    
    // Calculate rate (r) using central differences or simple forward/backward for ends
    const diffData = [];
    for (let i = 0; i < data.length - 1; i++) {
        const t1 = data[i].t;
        const CA1 = data[i].CA;
        const t2 = data[i+1].t;
        const CA2 = data[i+1].CA;
        
        const delta_t = t2 - t1;
        if(delta_t <= 0) continue;
        
        const delta_CA = CA2 - CA1;
        const r = -delta_CA / delta_t; 
        
        const CA_med = (CA1 + CA2) / 2;
        
        // Ensure values are positive for log
        if (r > 0 && CA_med > 0) {
            diffData.push({
                ln_r: Math.log(r),
                ln_CA: Math.log(CA_med)
            });
        }
    }
    
    const xs = diffData.map(d => d.ln_CA);
    const ys = diffData.map(d => d.ln_r);
    
    const reg = linearRegression(xs, ys);
    
    document.getElementById('diff-n').innerText = reg.slope.toFixed(3);
    document.getElementById('diff-lnk').innerText = reg.intercept.toFixed(3);
    const k_val = Math.exp(reg.intercept);
    document.getElementById('diff-k').innerText = k_val.toFixed(4);
    document.getElementById('diff-k-units').innerHTML = getKUnits(Math.round(reg.slope));
    document.getElementById('diff-r2').innerText = reg.r2.toFixed(4);
    
    plotDifferentialChart(xs, ys, reg);
}

function plotDifferentialChart(xs, ys, reg) {
    const ctx = document.getElementById('diff-chart').getContext('2d');
    if (state.diffChartItem) state.diffChartItem.destroy();
    
    const scatterData = xs.map((x, i) => ({ x: x, y: ys[i] }));
    const minX = Math.min(...xs) - 0.2;
    const maxX = Math.max(...xs) + 0.2;
    const lineData = [
        { x: minX, y: reg.slope * minX + reg.intercept },
        { x: maxX, y: reg.slope * maxX + reg.intercept }
    ];

    state.diffChartItem = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Datos Exp. ln(r) vs ln(Ca)',
                data: scatterData,
                backgroundColor: '#FF416C',
                pointRadius: 5
            }, {
                label: 'Ajuste Lineal',
                type: 'line',
                data: lineData,
                borderColor: '#FFD54F',
                borderWidth: 2,
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'ln(C_A)' } },
                y: { title: { display: true, text: 'ln(r)' } }
            }
        }
    });
}

function performIntegralAnalysis() {
    const { data } = state;
    if (data.length < 2) return;
    
    const ts = data.map(d => d.t);
    
    // Order 0: C_A = C_A0 - k*t  -> y = CA, x = t
    const y0 = data.map(d => d.CA);
    const reg0 = linearRegression(ts, y0);
    
    // Order 1: ln(C_A) = ln(C_A0) - k*t -> y = ln(CA), x = t
    // Filter CA > 0 to avoid log errors
    const ts1 = [], y1 = [];
    data.forEach(d => { if(d.CA > 0) { ts1.push(d.t); y1.push(Math.log(d.CA)); }});
    const reg1 = linearRegression(ts1, y1);
    
    // Order 2: 1/C_A = 1/C_A0 + k*t -> y = 1/CA, x = t
    const ts2 = [], y2 = [];
    data.forEach(d => { if(d.CA > 0) { ts2.push(d.t); y2.push(1 / d.CA); }});
    const reg2 = linearRegression(ts2, y2);
    
    // Order 3: 1/C_A^2 = 1/C_A0^2 + 2k*t -> y = 1/CA^2, x = t
    const ts3 = [], y3 = [];
    data.forEach(d => { if(d.CA > 0) { ts3.push(d.t); y3.push(1 / (d.CA * d.CA)); }});
    const reg3 = linearRegression(ts3, y3);
    
    document.getElementById('int-r2-0').innerText = reg0.r2.toFixed(4);
    document.getElementById('int-r2-1').innerText = reg1.r2.toFixed(4);
    document.getElementById('int-r2-2').innerText = reg2.r2.toFixed(4);
    document.getElementById('int-r2-3').innerText = reg3.r2.toFixed(4);
    
    let bestOrder = 0;
    let maxR2 = reg0.r2;
    let bReg = reg0;
    if(reg1.r2 > maxR2) { bestOrder = 1; maxR2 = reg1.r2; bReg = reg1; }
    if(reg2.r2 > maxR2) { bestOrder = 2; maxR2 = reg2.r2; bReg = reg2; }
    if(reg3.r2 > maxR2) { bestOrder = 3; maxR2 = reg3.r2; bReg = reg3; }
    
    let k_int = 0;
    if(bestOrder === 0) k_int = -bReg.slope;
    else if(bestOrder === 1) k_int = -bReg.slope;
    else if(bestOrder === 2) k_int = bReg.slope;
    else if(bestOrder === 3) k_int = bReg.slope / 2.0;

    document.getElementById('best-fit-order').innerText = `Orden ${bestOrder} (R² = ${maxR2.toFixed(4)})`;
    document.getElementById('int-k').innerText = k_int.toFixed(4);
    document.getElementById('int-k-units').innerHTML = getKUnits(bestOrder);

    // Store regs to plot when tabs swap
    state.integralRegs = [
        { x: ts, y: y0, reg: reg0, label: 'C_A' },
        { x: ts1, y: y1, reg: reg1, label: 'ln(C_A)' },
        { x: ts2, y: y2, reg: reg2, label: '1/C_A' },
        { x: ts3, y: y3, reg: reg3, label: '1/C_A^2' }
    ];
    
    plotIntegralChart(state.currentIntOrder);
}

function plotIntegralChart(orderIndex) {
    const ctx = document.getElementById('int-chart').getContext('2d');
    if (state.intChartItem) state.intChartItem.destroy();
    
    if(!state.integralRegs) return;
    
    const dataset = state.integralRegs[orderIndex];
    if(!dataset) return;
    
    const scatterData = dataset.x.map((x, i) => ({ x: x, y: dataset.y[i] }));
    const minX = Math.min(...dataset.x);
    const maxX = Math.max(...dataset.x);
    const lineData = [
        { x: minX, y: dataset.reg.slope * minX + dataset.reg.intercept },
        { x: maxX, y: dataset.reg.slope * maxX + dataset.reg.intercept }
    ];

    state.intChartItem = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: `Datos Exp. ${dataset.label} vs t`,
                data: scatterData,
                backgroundColor: '#00E676',
                pointRadius: 5
            }, {
                label: 'Ajuste Lineal',
                type: 'line',
                data: lineData,
                borderColor: '#FFD54F',
                borderWidth: 2,
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 't (min)' } },
                y: { title: { display: true, text: dataset.label } }
            }
        }
    });
}

// === Event Listeners ===

document.getElementById('btn-generate').addEventListener('click', generateProblem);

document.getElementById('btn-resolve').addEventListener('click', () => {
    document.getElementById('diff-panel').classList.remove('hidden');
    document.getElementById('int-panel').classList.remove('hidden');
    document.getElementById('btn-resolve').classList.add('hidden');
});

document.getElementById('btn-reveal').addEventListener('click', () => {
    const tp = state.trueParams;
    const el = document.getElementById('params-text');
    el.innerHTML = `<strong>n =</strong> ${tp.n}, <strong>k =</strong> ${tp.k.toFixed(3)} ${getKUnits(tp.n)}, <strong>C<sub>A0</sub> =</strong> ${tp.CA0.toFixed(2)} [mol/L]`;
});

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        state.currentIntOrder = parseInt(e.target.getAttribute('data-order'));
        plotIntegralChart(state.currentIntOrder);
    });
});

// Initialize on first load
// Render KaTeX globally initially
document.addEventListener("DOMContentLoaded", function() {
    generateProblem();
    
    // Initial math rendering handled by the deferred script in HTML
});
