#!/bin/bash
# ============================================================================
# DPS Voice Manager - Setup Script
# ============================================================================
# Instala dependencias, descarga modelos y configura el entorno.
#
# Uso:
#   chmod +x setup.sh
#   ./setup.sh
#
# Requisitos previos:
#   - Python 3.10+
#   - CUDA toolkit (para GPU)
#   - ffmpeg instalado
#   - HF_TOKEN configurado en .env o como variable de entorno
#     (requiere aceptar condiciones en HuggingFace para pyannote)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "============================================"
echo "  DPS Voice Manager - Setup"
echo "============================================"

# Verificar Python
PYTHON_CMD=""
if command -v python3.10 &>/dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.11 &>/dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Python 3.10+ no encontrado"
    exit 1
fi

echo "📦 Usando: $($PYTHON_CMD --version)"

# Verificar ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "⚠️  ffmpeg no encontrado. Instálalo con: sudo apt install ffmpeg"
fi

# Verificar HF_TOKEN
if [ -f "${SCRIPT_DIR}/.env" ]; then
    source "${SCRIPT_DIR}/.env"
fi

if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠️  HF_TOKEN no configurado."
    echo "   Pyannote requiere token de HuggingFace con acceso aceptado."
    echo "   1. Crea un token en: https://huggingface.co/settings/tokens"
    echo "   2. Acepta las condiciones en:"
    echo "      - https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "      - https://huggingface.co/pyannote/segmentation-3.0"
    echo "   3. Configura: echo 'HF_TOKEN=hf_xxx' > ${SCRIPT_DIR}/.env"
    echo ""
fi

# Crear entorno virtual
echo ""
echo "📁 Creando entorno virtual en ${VENV_DIR}..."
$PYTHON_CMD -m venv "$VENV_DIR"
source "${VENV_DIR}/bin/activate"

# Actualizar pip
echo "📦 Actualizando pip..."
pip install --upgrade pip wheel setuptools

# Instalar PyTorch con CUDA
echo ""
echo "🔥 Instalando PyTorch con CUDA..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar dependencias del proyecto
echo ""
echo "📦 Instalando dependencias..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# Crear directorios
echo ""
echo "📁 Creando directorios..."
mkdir -p "${SCRIPT_DIR}/temp_uploads"
mkdir -p "${SCRIPT_DIR}/models"
mkdir -p "${SCRIPT_DIR}/static"
mkdir -p "${SCRIPT_DIR}/logs"

# Descargar modelos (pre-caché)
echo ""
echo "🧠 Pre-descargando modelos..."
python -c "
import os
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')

print('  → Descargando SpeechBrain ECAPA-TDNN...')
try:
    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir='${SCRIPT_DIR}/models/ecapa_tdnn',
        run_opts={'device': 'cpu'}
    )
    print('  ✅ ECAPA-TDNN descargado')
except Exception as e:
    print(f'  ⚠️  Error descargando ECAPA-TDNN: {e}')

if os.environ.get('HF_TOKEN'):
    print('  → Descargando Pyannote diarization pipeline...')
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization-3.1',
            use_auth_token=os.environ['HF_TOKEN']
        )
        print('  ✅ Pyannote pipeline descargado')
    except Exception as e:
        print(f'  ⚠️  Error descargando Pyannote: {e}')
        print('     Verifica tu HF_TOKEN y que hayas aceptado las condiciones.')
else:
    print('  ⚠️  Saltando descarga de Pyannote (HF_TOKEN no configurado)')

print('')
print('✅ Setup completado')
"

echo ""
echo "============================================"
echo "  ✅ Setup completado"
echo "============================================"
echo ""
echo "Para iniciar el servicio:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  cd ${SCRIPT_DIR}"
echo "  python main.py"
echo ""
echo "El servicio estará en: http://localhost:3010"
echo ""
