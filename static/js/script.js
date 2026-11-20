document.addEventListener('DOMContentLoaded', () => {
    const $ = (sel, root = document) => root.querySelector(sel);
    const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
    const escapeHtml = (s) => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

    /* ---------- Pestañas ---------- */
    $$('.seg').forEach(btn => btn.addEventListener('click', () => {
        $$('.seg').forEach(b => b.classList.toggle('active', b === btn));
        $$('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `${btn.dataset.tab}-tab`));
    }));

    /* ---------- Navegación activa al hacer scroll ---------- */
    const navLinks = $$('.nav a');
    const sections = navLinks.map(a => $(a.getAttribute('href'))).filter(Boolean);
    window.addEventListener('scroll', () => {
        const y = window.scrollY + 120;
        let current = null;
        sections.forEach(s => { if (s.offsetTop <= y) current = s.id; });
        navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${current}`));
    }, { passive: true });

    /* ---------- Campos del formulario ---------- */
    // Cada campo tiene un <select> con niveles y un <input> numérico (modo exacto).
    // El input numérico siempre guarda el valor real que se envía al servidor.
    const fields = $$('.field').map(el => ({
        el,
        col: el.dataset.col,
        select: $('select', el),
        exact: $('input.exact', el),
        def: parseFloat($('select', el).dataset.default),
    }));
    const close = (a, b) => Math.abs(a - b) < 1e-4;
    let exampleLabel = null;  // valor real de C31 si se cargó un ejemplo sin modificar

    function refreshChanged(f) {
        f.el.classList.toggle('changed', !close(parseFloat(f.exact.value), f.def));
    }

    function syncSelect(f) {
        const v = parseFloat(f.exact.value);
        $$('option[data-custom]', f.select).forEach(o => o.remove());
        const match = [...f.select.options].find(o => close(parseFloat(o.value), v));
        if (match) {
            f.select.value = match.value;
        } else if (!Number.isNaN(v)) {
            const opt = new Option(`Personalizado (${v})`, v);
            opt.dataset.custom = '1';
            f.select.add(opt);
            f.select.value = opt.value;
        }
    }

    function setValue(f, v) {
        f.exact.value = v;
        f.exact.classList.remove('invalid');
        syncSelect(f);
        refreshChanged(f);
    }

    function userChanged() {
        if (exampleLabel !== null) {
            exampleLabel = null;
            showBanner('Modificaste la paciente de ejemplo: ya no se puede comparar con su resultado real.');
        }
    }

    fields.forEach(f => {
        f.select.addEventListener('change', () => {
            f.exact.value = f.select.value;
            f.exact.classList.remove('invalid');
            refreshChanged(f);
            userChanged();
        });
        f.exact.addEventListener('input', () => {
            const v = parseFloat(f.exact.value);
            f.exact.classList.toggle('invalid', Number.isNaN(v) || v < f.exact.min || v > f.exact.max);
            syncSelect(f);
            refreshChanged(f);
            userChanged();
        });
    });

    // Por defecto se escriben números; las variables categóricas usan lista.
    // El interruptor "Usar niveles en todo" cambia todos los campos a lista.
    $('#levels-mode')?.addEventListener('change', (e) => {
        fields.forEach(f => {
            const lista = e.target.checked || f.el.dataset.kind === 'select';
            f.select.hidden = !lista;
            f.exact.hidden = lista;
        });
    });

    const banner = $('#example-banner');
    function showBanner(html) {
        banner.innerHTML = html;
        banner.hidden = !html;
    }

    $('#reset-typical')?.addEventListener('click', () => {
        fields.forEach(f => setValue(f, f.def));
        exampleLabel = null;
        showBanner('');
        resetResult();
    });

    $('#load-example')?.addEventListener('click', async () => {
        try {
            const data = await (await fetch('/example')).json();
            fields.forEach(f => setValue(f, data.values[f.col]));
            exampleLabel = data.label;
            showBanner(`<i class="fas fa-shuffle"></i> Paciente real del estudio cargada. Su resultado real fue
                <strong>${data.label === 1 ? 'con restricción del crecimiento' : 'peso fetal normal'}</strong>: predice y compara.`);
            resetResult();
        } catch {
            showBanner('No se pudo cargar el ejemplo.');
        }
    });

    /* ---------- Predicción ---------- */
    const resultBox = $('#result');
    const modelName = (key) => $(`input[name="model_type"][value="${key}"]`)?.closest('label').querySelector('.mo-name').textContent.trim();

    function setResult(cls, html) {
        resultBox.className = `result ${cls}`;
        resultBox.innerHTML = html;
    }
    function resetResult() {
        setResult('empty', '<i class="fas fa-chart-pie"></i><p>Aquí aparecerá la predicción.</p>');
    }

    function collectValues() {
        const params = new URLSearchParams();
        const invalid = [];
        fields.forEach(f => {
            const v = parseFloat(f.exact.value);
            if (Number.isNaN(v) || v < f.exact.min || v > f.exact.max) invalid.push(f);
            params.append(f.col, f.exact.value);
        });
        invalid.forEach(f => f.exact.classList.add('invalid'));
        return { params, invalid };
    }

    async function predictWith(model, params) {
        const body = new URLSearchParams(params);
        body.set('model_type', model);
        const res = await fetch('/predict', { method: 'POST', body });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Error al procesar la solicitud');
        return data;
    }

    function checkInputs() {
        const { params, invalid } = collectValues();
        if (invalid.length) {
            setResult('error', `<strong><i class="fas fa-triangle-exclamation"></i> Revisa los campos marcados</strong>
                <p class="sub">Valores fuera de rango en: ${invalid.map(f => f.col).join(', ')}.</p>`);
            return null;
        }
        return params;
    }

    function matchNote(prediction) {
        if (exampleLabel === null) return '';
        return prediction === exampleLabel
            ? '<p class="sub"><i class="fas fa-circle-check"></i> Coincide con el resultado real de la paciente.</p>'
            : '<p class="sub"><i class="fas fa-circle-xmark"></i> No coincide con el resultado real de la paciente.</p>';
    }

    $('#individual-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const params = checkInputs();
        if (!params) return;
        const model = $('input[name="model_type"]:checked').value;
        const btn = $('#predict-btn');
        btn.disabled = true;
        setResult('loading', '<span class="spinner"></span> Calculando...');
        try {
            const data = await predictWith(model, params);
            const pos = data.prediction === 1;
            setResult(pos ? 'pos' : 'neg', `
                <div class="big"><i class="fas ${pos ? 'fa-circle-exclamation' : 'fa-circle-check'}"></i> ${escapeHtml(data.message)}</div>
                <p class="sub">Modelo: ${escapeHtml(modelName(model))}</p>${matchNote(data.prediction)}`);
        } catch (err) {
            setResult('error', `<strong><i class="fas fa-triangle-exclamation"></i> Error</strong><p class="sub">${escapeHtml(err.message)}</p>`);
        } finally {
            btn.disabled = false;
        }
    });

    $('#compare-btn')?.addEventListener('click', async () => {
        const params = checkInputs();
        if (!params) return;
        setResult('loading', '<span class="spinner"></span> Consultando los 4 modelos...');
        const models = $$('input[name="model_type"]').map(i => i.value);
        try {
            const results = await Promise.all(models.map(m => predictWith(m, params)));
            const items = models.map((m, i) => {
                const p = results[i].prediction;
                const ok = exampleLabel === null ? '' : (p === exampleLabel ? ' ✓' : ' ✗');
                return `<li><span>${escapeHtml(modelName(m))}</span><span class="${p ? 'p' : 'n'}">${p ? 'Restricción' : 'Normal'}${ok}</span></li>`;
            }).join('');
            const real = exampleLabel === null ? '' : `<p class="sub">Resultado real: ${exampleLabel ? 'Restricción' : 'Normal'}</p>`;
            setResult('', `<ul class="compare-list">${items}</ul>${real}`);
        } catch (err) {
            setResult('error', `<strong>Error</strong><p class="sub">${escapeHtml(err.message)}</p>`);
        }
    });

    /* ---------- Lote ---------- */
    const fileInput = $('#file');
    const dropzone = $('.dropzone');
    const fileInfo = $('.file-info');
    const defaultInfo = fileInfo?.textContent;

    fileInput?.addEventListener('change', () => {
        fileInfo.textContent = fileInput.files.length ? fileInput.files[0].name : defaultInfo;
    });
    if (dropzone) {
        ['dragenter', 'dragover'].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('drag'); }));
        ['dragleave', 'drop'].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('drag'); }));
        dropzone.addEventListener('drop', e => {
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                fileInput.dispatchEvent(new Event('change'));
            }
        });
    }

    // Matriz de confusión 2x2 dibujada con HTML: el color de cada celda depende de cuántos casos tiene.
    function matrizConfusion(m) {
        const maximo = Math.max(1, ...m.flat());
        const nombres = ['Normal', 'Restricción'];
        const celda = (valor, acierto) => {
            const alfa = (0.12 + 0.88 * valor / maximo).toFixed(2);
            const color = acierto ? `rgba(15, 143, 126, ${alfa})` : `rgba(225, 29, 72, ${alfa})`;
            const texto = valor / maximo > 0.55 ? '#fff' : 'var(--text)';
            return `<td style="background:${color};color:${texto}">${valor}</td>`;
        };
        const filas = m.map((fila, i) => `<tr><th scope="row">${nombres[i]}</th>${fila.map((v, j) => celda(v, i === j)).join('')}</tr>`).join('');
        return `<h4 class="cm-title">Matriz de confusión</h4>
            <table class="cm">
                <thead><tr><th></th><th colspan="2">Predicción</th></tr>
                       <tr><th>Real</th><th>${nombres[0]}</th><th>${nombres[1]}</th></tr></thead>
                <tbody>${filas}</tbody>
            </table>
            <p class="sub muted">Verde: aciertos · Rojo: errores</p>`;
    }

    $('#batch-prediction-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const box = $('#batch-result');
        const graph = $('#batch-graph');
        graph.innerHTML = '';
        if (!fileInput.files.length) {
            box.className = 'result error';
            box.innerHTML = '<strong>Elige un archivo primero.</strong>';
            return;
        }
        box.className = 'result loading';
        box.innerHTML = '<span class="spinner"></span> Procesando el lote...';
        try {
            const res = await fetch('/batch_predict', { method: 'POST', body: new FormData(e.target) });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Error al procesar el archivo');
            let html = `<div class="big">${data.rows} filas procesadas</div>
                <p class="sub">${data.positives} clasificadas con restricción del crecimiento.</p>`;
            if (data.accuracy) {
                html += `<p class="sub" style="margin-top:10px"><strong>Exactitud: ${escapeHtml(data.accuracy)}</strong></p>`;
            } else {
                html += '<p class="sub">El archivo no incluye C31, por eso no se calcula la exactitud.</p>';
            }
            box.className = 'result';
            box.innerHTML = html;
            if (data.confusion_matrix) graph.innerHTML = matrizConfusion(data.confusion_matrix);
        } catch (err) {
            box.className = 'result error';
            box.innerHTML = `<strong>Error</strong><p class="sub">${escapeHtml(err.message)}</p>`;
        }
    });
});
