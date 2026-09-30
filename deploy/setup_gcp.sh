#!/bin/bash
# ==============================================================================
# Tradi Copilot 24/7 - Script de Instalación Automatizada para GCP (Ubuntu 22.04)
# ==============================================================================
# Configura una VM e2-micro de Google Cloud (Always Free) para ejecutar Tradi 24/7.
# Incluye swap de 2GB, Python 3.11, entorno virtual, .env interactivo y systemd.
# ==============================================================================

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# Colores para salida informativa
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # Sin color

echo -e "${BLUE}====================================================================${NC}"
echo -e "${GREEN}   INICIANDO DESPLIEGUE AUTOMATIZADO DE TRADI COPILOT EN GCP        ${NC}"
echo -e "${BLUE}====================================================================${NC}"

# 1. Asegurar existencia del usuario 'ubuntu'
if ! id -u ubuntu >/dev/null 2>&1; then
    echo -e "${YELLOW}--> Creando usuario del sistema 'ubuntu'...${NC}"
    sudo useradd -m -s /bin/bash ubuntu
    sudo usermod -aG sudo ubuntu
fi

TARGET_DIR="/home/ubuntu/tradi"

# 2. Configurar 2GB de Swap para garantizar estabilidad en e2-micro (1GB RAM)
echo -e "${YELLOW}--> Verificando configuración de Swap (2GB para VM e2-micro)...${NC}"
if [ ! -f /swapfile ]; then
    echo -e "${YELLOW}    Creando archivo de intercambio /swapfile de 2GB...${NC}"
    sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    if ! grep -q '/swapfile' /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    echo -e "${GREEN}    Swap de 2GB configurado y habilitado correctamente.${NC}"
else
    echo -e "${GREEN}    El archivo /swapfile ya existe.${NC}"
    if ! swapon --show | grep -q '/swapfile'; then
        echo -e "${YELLOW}    Activando /swapfile existente...${NC}"
        sudo swapon /swapfile || true
    fi
fi

# 3. Actualizar sistema e instalar paquetes base de compilación
echo -e "${YELLOW}--> Actualizando paquetes de Ubuntu y repositorios base...${NC}"
sudo apt-get update -y
sudo apt-get upgrade -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"
sudo apt-get install -y software-properties-common curl git build-essential libfreetype6-dev libpng-dev

# 4. Instalar Python 3.11 y venv desde deadsnakes PPA
echo -e "${YELLOW}--> Instalando Python 3.11 desde ppa:deadsnakes/ppa...${NC}"
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# 5. Clonar o actualizar el repositorio en /home/ubuntu/tradi
echo -e "${YELLOW}--> Verificando código fuente en ${TARGET_DIR}...${NC}"
if [ -d "${TARGET_DIR}/.git" ]; then
    echo -e "${GREEN}    Repositorio existente detectado. Actualizando con git pull...${NC}"
    cd "${TARGET_DIR}"
    sudo -u ubuntu git pull origin main || true
else
    echo -e "${GREEN}    Clonando repositorio https://github.com/Amziisrae23/tradi.git...${NC}"
    sudo mkdir -p "${TARGET_DIR}"
    sudo chown -R ubuntu:ubuntu "/home/ubuntu"
    sudo -u ubuntu git clone https://github.com/Amziisrae23/tradi.git "${TARGET_DIR}"
    cd "${TARGET_DIR}"
fi

# 6. Crear entorno virtual con Python 3.11 e instalar dependencias
echo -e "${YELLOW}--> Configurando entorno virtual venv con Python 3.11...${NC}"
if [ ! -d "${TARGET_DIR}/venv" ]; then
    sudo -u ubuntu python3.11 -m venv "${TARGET_DIR}/venv"
fi

echo -e "${YELLOW}--> Actualizando pip e instalando dependencias desde requirements.txt...${NC}"
sudo -u ubuntu "${TARGET_DIR}/venv/bin/pip" install --upgrade pip setuptools wheel
sudo -u ubuntu "${TARGET_DIR}/venv/bin/pip" install -r "${TARGET_DIR}/requirements.txt"

# 7. Configuración interactiva de variables de entorno (.env)
echo ""
echo -e "${BLUE}====================================================================${NC}"
echo -e "${GREEN}          CONFIGURACIÓN DE VARIABLES DE ENTORNO (.env)              ${NC}"
echo -e "${BLUE}====================================================================${NC}"
echo "Por favor ingrese las credenciales del sistema (presione ENTER para dejar vacías las opcionales):"
echo ""

read -r -p "1. BITUNIX_API_KEY (deja vacío para modo simulación/lectura pública): " BITUNIX_API_KEY || BITUNIX_API_KEY=""
read -r -p "2. BITUNIX_API_SECRET (deja vacío para modo simulación): " BITUNIX_API_SECRET || BITUNIX_API_SECRET=""
read -r -p "3. TELEGRAM_BOT_TOKEN (Token de @BotFather): " TELEGRAM_BOT_TOKEN || TELEGRAM_BOT_TOKEN=""
read -r -p "4. TELEGRAM_CHAT_ID (ID numérico de @userinfobot): " TELEGRAM_CHAT_ID || TELEGRAM_CHAT_ID=""
read -r -p "5. GEMINI_API_KEY (API Key gratuita de Google AI Studio): " GEMINI_API_KEY || GEMINI_API_KEY=""

ENV_FILE="${TARGET_DIR}/.env"
echo -e "${YELLOW}--> Generando archivo seguro ${ENV_FILE}...${NC}"

cat <<EOF | sudo -u ubuntu tee "${ENV_FILE}" > /dev/null
# Bitunix API Credentials
BITUNIX_API_KEY=${BITUNIX_API_KEY}
BITUNIX_API_SECRET=${BITUNIX_API_SECRET}
BITUNIX_REST_URL=https://fapi.bitunix.com
BITUNIX_WS_URL=wss://fapi.bitunix.com/public/

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}

# Google Gemini 2.0 Flash AI Analyst
GEMINI_API_KEY=${GEMINI_API_KEY}

# Cloud Web Service & Healthcheck Port
PORT=8080
EOF

sudo chmod 600 "${ENV_FILE}"
sudo chown ubuntu:ubuntu "${ENV_FILE}"
echo -e "${GREEN}    Archivo .env generado y protegido con permisos 600.${NC}"

# 8. Instalar y habilitar el servicio systemd
echo -e "${YELLOW}--> Instalando y configurando servicio systemd tradi.service...${NC}"
sudo cp "${TARGET_DIR}/deploy/tradi.service" /etc/systemd/system/tradi.service
sudo chmod 644 /etc/systemd/system/tradi.service
sudo chown -R ubuntu:ubuntu "${TARGET_DIR}"

sudo systemctl daemon-reload
sudo systemctl enable tradi.service
sudo systemctl restart tradi.service

# 9. Verificación de estado e información de administración
echo ""
echo -e "${BLUE}====================================================================${NC}"
if sudo systemctl is-active --quiet tradi.service; then
    echo -e "${GREEN}✅ ¡TRADI COPILOT SE ENCUENTRA ACTIVO Y EJECUTÁNDOSE 24/7 EN GCP!    ${NC}"
else
    echo -e "${RED}⚠️ El servicio no arrancó inmediatamente. Revisa los logs con:      ${NC}"
    echo -e "${RED}   sudo journalctl -u tradi.service -n 50 --no-pager                 ${NC}"
fi
echo -e "${BLUE}====================================================================${NC}"
echo -e "Comandos útiles para administración del bot:"
echo -e " - Ver estado:        ${YELLOW}sudo systemctl status tradi.service${NC}"
echo -e " - Ver logs en vivo:  ${YELLOW}sudo journalctl -u tradi.service -f${NC}"
echo -e " - Reiniciar bot:     ${YELLOW}sudo systemctl restart tradi.service${NC}"
echo -e " - Detener bot:       ${YELLOW}sudo systemctl stop tradi.service${NC}"
echo -e " - Probar endpoint:   ${YELLOW}curl http://localhost:8080/health${NC}"
echo ""
