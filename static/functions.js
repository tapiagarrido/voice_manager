let selectedFiles = [];
const SPEAKER_COLORS = [
    '#8b5cf6', '#ef4444', '#3b82f6', '#10b981', '#f59e0b',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
];

// ===== TABS =====
function showTab(tab) {
    ['register', 'persons', 'test'].forEach(t => {
        document.getElementById('panel-' + t).classList.toggle('hidden', t !== tab);
        const tabBtn = document.getElementById('tab-' + t);
        tabBtn.classList.toggle('text-purple-600', t === tab);
        tabBtn.classList.toggle('border-purple-600', t === tab);
        tabBtn.classList.toggle('border-b-2', t === tab);
        tabBtn.classList.toggle('text-gray-500', t !== tab);
    });
    if (tab === 'persons') loadPersons();
}

// ===== HEALTH CHECK =====
async function checkHealth() {
    try {
        const r = await fetch('/api/health');
        const data = await r.json();
        const s = data.subsystems;
        const parts = [];
        parts.push(s.elasticsearch ? '✅ ES' : '❌ ES');
        parts.push(s.voice_embedding ? '✅ Embedding' : '❌ Embedding');
        parts.push(s.diarization ? '✅ Diarización' : '⚠️ Diarización');
        parts.push(`📊 ${data.voice_bank.total_persons} voces registradas`);
        document.getElementById('healthStatus').innerHTML = parts.join(' • ');
    } catch (e) {
        document.getElementById('healthStatus').innerHTML = '❌ Servicio no disponible';
    }
}

// ===== DROPZONE =====
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
dropzone.addEventListener('click', (e) => {
    if (e.target !== fileInput) fileInput.click();
});
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); handleFiles(e.dataTransfer.files); });

function handleFiles(files) {
    for (const file of files) {
        if (file.type.startsWith('audio/') && selectedFiles.length < 10) {
            selectedFiles.push(file);
        }
    }
    updateAudioPreview();
}

function updateAudioPreview() {
    const preview = document.getElementById('audioPreview');
    const count = document.getElementById('audioCount');

    preview.innerHTML = selectedFiles.map((file, i) => `
                <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div class="flex items-center gap-3">
                        <span class="text-purple-500">🎵</span>
                        <div>
                            <p class="text-sm font-medium">${file.name}</p>
                            <p class="text-xs text-gray-400">${(file.size / 1024 / 1024).toFixed(1)} MB</p>
                        </div>
                    </div>
                    <div class="flex items-center gap-2">
                        <audio controls class="h-8" style="max-width: 200px">
                            <source src="${URL.createObjectURL(file)}">
                        </audio>
                        <button type="button" onclick="removeFile(${i})" class="text-red-500 hover:text-red-700 text-lg">×</button>
                    </div>
                </div>
            `).join('');

    const n = selectedFiles.length;
    let color = n < 1 ? 'text-red-500' : n < 3 ? 'text-yellow-600' : 'text-green-600';
    count.innerHTML = `<span class="${color}">${n} clip${n !== 1 ? 's' : ''} seleccionado${n !== 1 ? 's' : ''}</span>`;
    document.getElementById('submitBtn').disabled = n < 1;
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateAudioPreview();
}

// ===== REGISTRO =====
async function submitRegistration(e) {
    e.preventDefault();
    if (selectedFiles.length < 1) { alert('Se requiere al menos 1 clip'); return; }

    const formData = new FormData();
    formData.append('name', document.getElementById('personName').value);
    formData.append('aliases', document.getElementById('personAliases').value);
    formData.append('role', document.getElementById('personRole').value);
    selectedFiles.forEach(file => formData.append('audio_files', file));

    document.getElementById('progressContainer').classList.remove('hidden');
    document.getElementById('submitBtn').disabled = true;

    try {
        const response = await fetch('/api/register', { method: 'POST', body: formData });
        const result = await response.json();

        const container = document.getElementById('resultContainer');
        container.classList.remove('hidden');

        if (response.ok && result.success) {
            container.className = 'mt-4 p-4 rounded-lg bg-green-100 text-green-800';
            container.innerHTML = `
                        <p class="font-medium">✅ ${result.message}</p>
                        <p class="text-sm mt-1">ID: ${result.person_id} • Confianza: ${(result.avg_confidence * 100).toFixed(0)}%</p>
                        ${result.errors.length ? `<p class="text-sm mt-1 text-yellow-700">⚠️ Warnings: ${result.errors.join(', ')}</p>` : ''}
                    `;
            document.getElementById('registerForm').reset();
            selectedFiles = [];
            updateAudioPreview();
            checkHealth();
        } else {
            container.className = 'mt-4 p-4 rounded-lg bg-red-100 text-red-800';
            container.innerHTML = `<p class="font-medium">❌ ${result.detail || result.error || 'Error desconocido'}</p>`;
        }
    } catch (error) {
        alert('Error de conexión: ' + error.message);
    } finally {
        document.getElementById('progressContainer').classList.add('hidden');
        document.getElementById('progressBar').style.width = '0%';
        document.getElementById('submitBtn').disabled = false;
    }
}

// ===== VOCES REGISTRADAS =====
let allPersons = [];

async function loadPersons() {
    const list = document.getElementById('personsList');
    list.innerHTML = '<p class="text-gray-500 text-center py-8">Cargando...</p>';
    try {
        const r = await fetch('/api/persons');
        allPersons = await r.json();
        renderPersons(allPersons);
    } catch (e) {
        list.innerHTML = '<p class="text-red-500 text-center py-8">Error cargando voces</p>';
    }
}

const ROLE_EMOJIS = {
    'locutor': '🎙️', 'periodista': '📰', 'comentarista': '💬',
    'conductor': '📺', 'corresponsal': '🌍', 'invitado': '🎤',
    'analista': '🔍', 'reportero': '📡', 'otro': '👤'
};

function renderPersons(persons) {
    const list = document.getElementById('personsList');
    if (!persons.length) {
        list.innerHTML = '<p class="text-gray-500 text-center py-8">No hay voces registradas</p>';
        return;
    }

    list.innerHTML = persons.map(p => `
                <div class="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100">
                    <div class="flex items-center gap-4">
                        <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center text-xl">
                            ${ROLE_EMOJIS[p.role] || '🎙️'}
                        </div>
                        <div>
                            <p class="font-medium text-gray-800">${p.name}</p>
                            <div class="flex items-center gap-2 text-sm text-gray-500">
                                <span class="speaker-tag bg-purple-100 text-purple-700">${p.role || 'locutor'}</span>
                                ${p.aliases && p.aliases.length ? ` <span>• ${p.aliases.join(', ')}</span>` : ''}
                            </div>
                            <p class="text-xs text-gray-400">${p.sample_count || 0} clips • ${(p.total_speech_duration || 0).toFixed(0)}s habla • Consistencia: ${((p.avg_confidence || 0) * 100).toFixed(0)}%</p>
                        </div>
                    </div>
                    <div class="flex gap-2">
                        <button onclick="openEditModal('${p.person_id}', '${p.name.replace(/'/g, "\\'")}', '${(p.aliases || []).join(', ').replace(/'/g, "\\'")}', '${p.role || 'locutor'}')" 
                                class="text-blue-500 hover:text-blue-700 p-2" title="Editar">✏️</button>
                        <button onclick="deletePerson('${p.person_id}', '${p.name.replace(/'/g, "\\'")}')" 
                                class="text-red-500 hover:text-red-700 p-2" title="Eliminar">🗑️</button>
                    </div>
                </div>
            `).join('');
}

function filterPersons(query) {
    const q = query.toLowerCase();
    const filtered = allPersons.filter(p =>
        p.name.toLowerCase().includes(q) ||
        (p.role || '').toLowerCase().includes(q) ||
        (p.aliases && p.aliases.some(a => a.toLowerCase().includes(q)))
    );
    renderPersons(filtered);
}

async function deletePerson(id, name) {
    if (!confirm(`¿Eliminar la voz de "${name}"?`)) return;
    try {
        const r = await fetch(`/api/persons/${id}`, { method: 'DELETE' });
        if (r.ok) { loadPersons(); checkHealth(); }
        else alert('Error eliminando voz');
    } catch (e) { alert('Error de conexión'); }
}

// ===== EDITAR =====
function openEditModal(personId, name, aliases, role) {
    document.getElementById('editPersonId').value = personId;
    document.getElementById('editName').value = name;
    document.getElementById('editAliases').value = aliases;
    document.getElementById('editRole').value = role;
    document.getElementById('editModal').classList.remove('hidden');
}

function closeEditModal() {
    document.getElementById('editModal').classList.add('hidden');
}

async function submitEdit(e) {
    e.preventDefault();
    const personId = document.getElementById('editPersonId').value;
    const name = document.getElementById('editName').value.trim();
    const aliasesStr = document.getElementById('editAliases').value.trim();
    const aliases = aliasesStr ? aliasesStr.split(',').map(a => a.trim()).filter(a => a) : [];
    const role = document.getElementById('editRole').value;

    document.getElementById('editSubmitBtn').disabled = true;
    try {
        const r = await fetch(`/api/persons/${personId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, aliases, role })
        });
        if (r.ok) { closeEditModal(); loadPersons(); }
        else { const res = await r.json(); alert('Error: ' + (res.detail || 'Error actualizando')); }
    } catch (e) { alert('Error de conexión'); }
    finally { document.getElementById('editSubmitBtn').disabled = false; }
}

// ===== TEST DIARIZACIÓN =====
async function runTest() {
    const fileInput = document.getElementById('testAudioInput');
    if (!fileInput.files.length) { alert('Selecciona un archivo de audio'); return; }

    const mode = document.getElementById('testMode').value;
    const file = fileInput.files[0];

    document.getElementById('testBtn').disabled = true;
    document.getElementById('testLoading').classList.remove('hidden');
    document.getElementById('testResults').classList.add('hidden');

    try {
        const formData = new FormData();
        formData.append('audio', file);

        let url, response;

        if (mode === 'diarize') {
            const numSpeakers = document.getElementById('testNumSpeakers').value;
            const threshold = document.getElementById('testThreshold').value;
            const identify = document.getElementById('testIdentify').checked;

            if (numSpeakers) formData.append('num_speakers', numSpeakers);
            if (threshold) formData.append('threshold', threshold);
            formData.append('identify', identify);

            response = await fetch('/api/diarize', { method: 'POST', body: formData });
        } else {
            response = await fetch('/api/identify-speaker', { method: 'POST', body: formData });
        }

        const result = await response.json();

        if (!response.ok) {
            alert('Error: ' + (result.detail || 'Error procesando'));
            return;
        }

        document.getElementById('testResults').classList.remove('hidden');

        if (mode === 'diarize') {
            renderDiarizationResults(result);
        } else {
            renderIdentifyResults(result);
        }

    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        document.getElementById('testBtn').disabled = false;
        document.getElementById('testLoading').classList.add('hidden');
    }
}

function renderDiarizationResults(result) {
    // Summary
    const summary = document.getElementById('testSummary');
    summary.innerHTML = `
                <div class="grid grid-cols-3 gap-4 text-center">
                    <div>
                        <p class="text-2xl font-bold text-purple-600">${result.total_speakers}</p>
                        <p class="text-sm text-gray-600">Speakers</p>
                    </div>
                    <div>
                        <p class="text-2xl font-bold text-green-600">${result.identified_speakers}</p>
                        <p class="text-sm text-gray-600">Identificados</p>
                    </div>
                    <div>
                        <p class="text-2xl font-bold text-orange-500">${result.unidentified_speakers}</p>
                        <p class="text-sm text-gray-600">Desconocidos</p>
                    </div>
                </div>
                <div class="mt-3 flex flex-wrap gap-2">
                    ${Object.entries(result.speakers).map(([name, info], i) => `
                        <span class="speaker-tag text-white" style="background-color: ${SPEAKER_COLORS[i % SPEAKER_COLORS.length]}">
                            ${info.identified ? '✅' : '❓'} ${name} (${info.total_time}s)
                            ${info.similarity ? ' • ' + (info.similarity * 100).toFixed(0) + '%' : ''}
                        </span>
                    `).join('')}
                </div>
            `;

    // Timeline
    if (result.segments.length > 0) {
        const maxTime = Math.max(...result.segments.map(s => s.end));
        const speakers = [...new Set(result.segments.map(s => s.speaker))];
        const speakerColors = {};
        speakers.forEach((s, i) => speakerColors[s] = SPEAKER_COLORS[i % SPEAKER_COLORS.length]);

        const timeline = document.getElementById('testTimeline');
        timeline.innerHTML = `
                    <p class="text-sm font-medium text-gray-700 mb-2">Timeline (${maxTime.toFixed(1)}s)</p>
                    <div class="relative bg-gray-200 rounded h-10 overflow-hidden">
                        ${result.segments.map(s => {
            const left = (s.start / maxTime * 100).toFixed(2);
            const width = ((s.end - s.start) / maxTime * 100).toFixed(2);
            return `<div class="timeline-segment" title="${s.speaker}: ${s.start.toFixed(1)}s - ${s.end.toFixed(1)}s"
                                         style="left: ${left}%; width: ${width}%; background-color: ${speakerColors[s.speaker]}; top: 5px; height: 20px;"></div>`;
        }).join('')}
                    </div>
                    <div class="flex justify-between text-xs text-gray-400 mt-1">
                        <span>0:00</span>
                        <span>${Math.floor(maxTime / 60)}:${String(Math.floor(maxTime % 60)).padStart(2, '0')}</span>
                    </div>
                `;
    }

    // Segments table
    const segsDiv = document.getElementById('testSegments');
    segsDiv.innerHTML = `
                <p class="text-sm font-medium text-gray-700 mb-2">Segmentos (${result.segments.length})</p>
                <div class="max-h-96 overflow-y-auto">
                    <table class="w-full text-sm">
                        <thead class="bg-gray-100 sticky top-0">
                            <tr>
                                <th class="px-3 py-2 text-left">Inicio</th>
                                <th class="px-3 py-2 text-left">Fin</th>
                                <th class="px-3 py-2 text-left">Duración</th>
                                <th class="px-3 py-2 text-left">Speaker</th>
                                <th class="px-3 py-2 text-left">Confianza</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${result.segments.map((s, i) => {
        const speakers = [...new Set(result.segments.map(x => x.speaker))];
        const colorIdx = speakers.indexOf(s.speaker);
        return `<tr class="border-b hover:bg-gray-50">
                                    <td class="px-3 py-1">${formatTime(s.start)}</td>
                                    <td class="px-3 py-1">${formatTime(s.end)}</td>
                                    <td class="px-3 py-1">${s.duration.toFixed(1)}s</td>
                                    <td class="px-3 py-1">
                                        <span class="speaker-tag text-white" style="background-color: ${SPEAKER_COLORS[colorIdx % SPEAKER_COLORS.length]}">${s.speaker}</span>
                                        ${s.role ? `<span class="text-xs text-gray-400">${s.role}</span>` : ''}
                                    </td>
                                    <td class="px-3 py-1">${s.confidence ? (s.confidence * 100).toFixed(0) + '%' : '-'}</td>
                                </tr>`;
    }).join('')}
                        </tbody>
                    </table>
                </div>
            `;
}

function renderIdentifyResults(result) {
    const summary = document.getElementById('testSummary');
    if (result.matches && result.matches.length > 0) {
        summary.innerHTML = `
                    <p class="font-medium text-green-700 mb-3">✅ ${result.matches.length} coincidencia(s) encontrada(s)</p>
                    ${result.matches.map((m, i) => `
                        <div class="flex items-center justify-between p-3 ${i === 0 ? 'bg-green-50' : 'bg-gray-50'} rounded-lg mb-2">
                            <div>
                                <p class="font-medium">${m.name} <span class="text-sm text-gray-500">${m.role || ''}</span></p>
                                <p class="text-xs text-gray-400">ID: ${m.person_id}</p>
                            </div>
                            <div class="text-right">
                                <p class="text-lg font-bold ${m.similarity >= 0.7 ? 'text-green-600' : 'text-yellow-600'}">
                                    ${(m.similarity * 100).toFixed(1)}%
                                </p>
                                <p class="text-xs text-gray-400">similitud</p>
                            </div>
                        </div>
                    `).join('')}
                `;
    } else {
        summary.innerHTML = '<p class="text-yellow-700">❓ No se encontraron coincidencias en el banco de voces</p>';
    }
    document.getElementById('testTimeline').innerHTML = '';
    document.getElementById('testSegments').innerHTML = '';
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1);
    return `${m}:${s.padStart(4, '0')}`;
}

// ===== INIT =====
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeEditModal(); });
updateAudioPreview();
checkHealth();
