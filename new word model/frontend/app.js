/**
 * Handwriting Docs — In-Place Real-Time Recognition & Substitution Engine
 * 
 * Intercepts handwritten stroke input over the document page, sends temporal
 * coordinate sequences to the BiLSTM-CTC backend, and seamlessly substitutes
 * the ink with recognized text directly in the document flow.
 */

// DOM Elements
const pageContainer = document.getElementById('pageContainer');
const docEditor = document.getElementById('docEditor');
const inkCanvas = document.getElementById('inkCanvas');
const ctx = inkCanvas.getContext('2d');

const aiStatusPill = document.getElementById('aiStatusPill');
const aiStatusText = document.getElementById('aiStatusText');
const latencyValue = document.getElementById('latencyValue');
const docStats = document.getElementById('docStats');
const saveStatus = document.getElementById('saveStatus');
const docTitle = document.getElementById('docTitle');
const insertionIndicator = document.getElementById('insertionIndicator');

// Toolbar Controls
const btnHandwritingMode = document.getElementById('btnHandwritingMode');
const btnClearInk = document.getElementById('btnClearInk');
const btnRecognizeNow = document.getElementById('btnRecognizeNow');
const btnCopyAll = document.getElementById('btnCopyAll');
const btnSampleInsert = document.getElementById('btnSampleInsert');
const btnPrint = document.getElementById('btnPrint');
const colorDots = document.querySelectorAll('.color-dot');
const formatBlockSelect = document.getElementById('formatBlock');

// State
let isHandwritingActive = true;
let isDrawing = false;
let strokes = [];            // Array of strokes: [ [ {x, y, time}, ... ], ... ]
let currentStroke = [];
let strokeStartTime = 0;
let inkColor = '#1e293b';
let debounceTimer = null;
let lastPointerPos = { x: 0, y: 0 };
let activeTargetNode = null; // Node where substituted text will be placed

// ─── 1. Canvas Setup & High-DPI Scaling ──────────────────────────────────────

function resizeInkCanvas() {
    const rect = pageContainer.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    inkCanvas.width = rect.width * dpr;
    inkCanvas.height = rect.height * dpr;
    
    ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform
    ctx.scale(dpr, dpr);
    redrawInk();
}

window.addEventListener('resize', resizeInkCanvas);
window.addEventListener('load', resizeInkCanvas);
setTimeout(resizeInkCanvas, 100);

function getCanvasPoint(e) {
    const rect = pageContainer.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        time: (performance.now() - strokeStartTime) / 1000.0
    };
}

// ─── 2. Handwriting Stroke Recording ────────────────────────────────────────

inkCanvas.addEventListener('pointerdown', (e) => {
    if (!isHandwritingActive) return;
    
    inkCanvas.setPointerCapture(e.pointerId);
    isDrawing = true;

    if (strokes.length === 0 && currentStroke.length === 0) {
        strokeStartTime = performance.now();
        // Determine insertion target on document where user began writing
        findInsertionTarget(e.clientX, e.clientY);
    }

    const pt = getCanvasPoint(e);
    lastPointerPos = { x: e.clientX, y: e.clientY };
    currentStroke = [pt];

    ctx.beginPath();
    ctx.strokeStyle = inkColor;
    ctx.lineWidth = 3.0;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.moveTo(pt.x, pt.y);
    ctx.lineTo(pt.x, pt.y);
    ctx.stroke();

    updateStatus('Writing...', 'active');
});

inkCanvas.addEventListener('pointermove', (e) => {
    if (!isDrawing || !isHandwritingActive) return;

    const pt = getCanvasPoint(e);
    currentStroke.push(pt);

    if (currentStroke.length > 1) {
        const p1 = currentStroke[currentStroke.length - 2];
        const p2 = currentStroke[currentStroke.length - 1];
        ctx.beginPath();
        ctx.strokeStyle = inkColor;
        ctx.lineWidth = 3.0;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
    }
});

function handleStrokeEnd() {
    if (!isDrawing) return;
    isDrawing = false;

    if (currentStroke.length > 0) {
        strokes.push(currentStroke);
        currentStroke = [];
    }

    // Auto-recognize & substitute in-place after user pauses writing (~480ms)
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        substituteHandwritingInPlace();
    }, 480);
}

inkCanvas.addEventListener('pointerup', handleStrokeEnd);
inkCanvas.addEventListener('pointercancel', handleStrokeEnd);

function redrawInk() {
    const rect = pageContainer.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);

    ctx.strokeStyle = inkColor;
    ctx.lineWidth = 3.0;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const stroke of strokes) {
        if (stroke.length === 0) continue;
        ctx.beginPath();
        ctx.moveTo(stroke[0].x, stroke[0].y);
        for (let i = 1; i < stroke.length; i++) {
            ctx.lineTo(stroke[i].x, stroke[i].y);
        }
        ctx.stroke();
    }
}

// ─── 3. In-Place Target Finding ─────────────────────────────────────────────

function findInsertionTarget(clientX, clientY) {
    // Check elements in docEditor under coordinates
    const elUnder = document.elementFromPoint(clientX, clientY);
    if (elUnder && docEditor.contains(elUnder)) {
        activeTargetNode = elUnder;
    } else {
        // Find nearest paragraph by vertical proximity
        const paragraphs = docEditor.querySelectorAll('p, h1, h2, h3, div');
        let closestP = null;
        let minDistance = Infinity;

        paragraphs.forEach(p => {
            const r = p.getBoundingClientRect();
            const dist = Math.abs(clientY - (r.top + r.height / 2));
            if (dist < minDistance) {
                minDistance = dist;
                closestP = p;
            }
        });

        activeTargetNode = closestP || docEditor.lastElementChild || docEditor;
    }
}

// ─── 4. In-Place Substitution Execution ──────────────────────────────────────

async function substituteHandwritingInPlace() {
    if (strokes.length === 0) return;

    const strokePayload = strokes;
    updateStatus('Recognizing in-place...', 'recognizing');

    const t0 = performance.now();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strokes: strokePayload })
        });

        const data = await response.json();
        const latency = Math.round(performance.now() - t0);
        latencyValue.textContent = data.latency_ms || latency;

        const recognizedText = (data.text || '').trim();

        if (recognizedText.length > 0) {
            // Animate ink fade out
            fadeAndClearInk();

            // Insert recognized text in-place at the active target position
            insertTextInPlace(recognizedText);

            updateStatus(`Substituted: "${recognizedText}"`, 'success');
            setTimeout(() => updateStatus('Live In-Place Recognition Ready', 'ready'), 2500);
        } else {
            updateStatus('No clear text recognized', 'ready');
        }

    } catch (err) {
        console.error('In-place recognition error:', err);
        updateStatus('Recognition failed', 'ready');
    } finally {
        strokes = [];
        currentStroke = [];
    }
}

function insertTextInPlace(text) {
    // Ensure document has focus
    docEditor.focus();

    // Create a highlighted span with substitution glow animation
    const span = document.createElement('span');
    span.className = 'substituted-text-glow';
    span.textContent = text + ' ';

    // Target node insertion
    if (activeTargetNode && docEditor.contains(activeTargetNode)) {
        if (activeTargetNode === docEditor) {
            const p = document.createElement('p');
            p.appendChild(span);
            docEditor.appendChild(p);
        } else if (activeTargetNode.nodeName === 'P' && activeTargetNode.innerHTML === '<br>') {
            activeTargetNode.innerHTML = '';
            activeTargetNode.appendChild(span);
        } else {
            activeTargetNode.appendChild(span);
        }
    } else {
        // Append to last element or create new paragraph
        let lastP = docEditor.lastElementChild;
        if (!lastP || lastP.nodeName !== 'P') {
            lastP = document.createElement('p');
            docEditor.appendChild(lastP);
        }
        lastP.appendChild(span);
    }

    // Place cursor right after the substituted text
    const range = document.createRange();
    const sel = window.getSelection();
    range.setStartAfter(span);
    range.collapse(true);
    sel.removeAllRanges();
    sel.addRange(range);

    updateDocStats();
    saveDocument();
}

function fadeAndClearInk() {
    let opacity = 1.0;
    const fadeInterval = setInterval(() => {
        opacity -= 0.15;
        if (opacity <= 0) {
            clearInterval(fadeInterval);
            ctx.clearRect(0, 0, inkCanvas.width, inkCanvas.height);
        } else {
            redrawInk();
            ctx.globalAlpha = opacity;
        }
    }, 25);
}

// ─── 5. UI Controls & Formatting ────────────────────────────────────────────

function updateStatus(text, state) {
    aiStatusText.textContent = text;
    aiStatusPill.className = 'status-pill ' + (state || '');
}

function updateDocStats() {
    const text = docEditor.innerText || '';
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const chars = text.length;
    docStats.textContent = `Words: ${words} | Characters: ${chars}`;
}

docEditor.addEventListener('input', () => {
    updateDocStats();
    saveDocument();
});

function saveDocument() {
    saveStatus.textContent = 'Saving...';
    localStorage.setItem('gdocs_hw_content', docEditor.innerHTML);
    localStorage.setItem('gdocs_hw_title', docTitle.value);
    setTimeout(() => {
        saveStatus.textContent = 'Saved to browser';
    }, 300);
}

// Load saved document on startup
window.addEventListener('DOMContentLoaded', () => {
    const saved = localStorage.getItem('gdocs_hw_content');
    const savedTitle = localStorage.getItem('gdocs_hw_title');
    if (saved) docEditor.innerHTML = saved;
    if (savedTitle) docTitle.value = savedTitle;
    updateDocStats();
});

// Handwriting Mode Toggle
btnHandwritingMode.addEventListener('click', () => {
    isHandwritingActive = !isHandwritingActive;
    if (isHandwritingActive) {
        btnHandwritingMode.classList.add('active');
        btnHandwritingMode.querySelector('span').textContent = 'Handwriting Active';
        inkCanvas.style.pointerEvents = 'auto';
    } else {
        btnHandwritingMode.classList.remove('active');
        btnHandwritingMode.querySelector('span').textContent = 'Typing Only';
        inkCanvas.style.pointerEvents = 'none';
        ctx.clearRect(0, 0, inkCanvas.width, inkCanvas.height);
    }
});

// Color Picker
colorDots.forEach(dot => {
    dot.addEventListener('click', () => {
        colorDots.forEach(d => d.classList.remove('active'));
        dot.classList.add('active');
        inkColor = dot.getAttribute('data-color');
    });
});

// Clear Ink
btnClearInk.addEventListener('click', () => {
    strokes = [];
    currentStroke = [];
    ctx.clearRect(0, 0, inkCanvas.width, inkCanvas.height);
    updateStatus('Ink cleared', 'ready');
});

// Force Recognize Now
btnRecognizeNow.addEventListener('click', () => {
    clearTimeout(debounceTimer);
    substituteHandwritingInPlace();
});

// Copy Document Text
btnCopyAll.addEventListener('click', () => {
    const text = docEditor.innerText;
    navigator.clipboard.writeText(text);
    btnCopyAll.innerHTML = `✓ Copied!`;
    setTimeout(() => {
        btnCopyAll.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            Copy Text`;
    }, 1500);
});

// Rich Text Formatting Tools
document.getElementById('btnBold').addEventListener('click', () => document.execCommand('bold', false, null));
document.getElementById('btnItalic').addEventListener('click', () => document.execCommand('italic', false, null));
document.getElementById('btnUnderline').addEventListener('click', () => document.execCommand('underline', false, null));
document.getElementById('btnDocUndo').addEventListener('click', () => document.execCommand('undo', false, null));
document.getElementById('btnDocRedo').addEventListener('click', () => document.execCommand('redo', false, null));
btnPrint.addEventListener('click', () => window.print());

formatBlockSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    document.execCommand('formatBlock', false, val);
});

// Sample Insert Simulation
btnSampleInsert.addEventListener('click', async () => {
    updateStatus('Loading IAM sample strokes...', 'recognizing');
    try {
        const res = await fetch('/api/sample');
        const data = await res.json();

        if (data.strokes && data.strokes.length > 0) {
            // Position sample inside document view
            const rect = pageContainer.getBoundingClientRect();
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

            for (const s of data.strokes) {
                for (const p of s) {
                    if (p.x < minX) minX = p.x;
                    if (p.x > maxX) maxX = p.x;
                    if (p.y < minY) minY = p.y;
                    if (p.y > maxY) maxY = p.y;
                }
            }

            const strokeW = maxX - minX || 1;
            const strokeH = maxY - minY || 1;
            const scale = Math.min((rect.width - 160) / strokeW, 90 / strokeH);
            const offsetX = 80;
            const offsetY = 180;

            strokes = data.strokes.map(s => s.map(p => ({
                x: (p.x - minX) * scale + offsetX,
                y: (p.y - minY) * scale + offsetY,
                time: p.time
            })));

            redrawInk();
            setTimeout(() => {
                substituteHandwritingInPlace();
            }, 600);
        }
    } catch (e) {
        console.error(e);
        updateStatus('Failed to load sample', 'ready');
    }
});
