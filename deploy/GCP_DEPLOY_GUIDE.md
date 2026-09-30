# Guía Definitiva de Despliegue 24/7 en Google Cloud Platform (GCP Always Free)
## Tradi Copilot — Analista Cuantitativo Senior con Gemini 2.0 Flash

Esta guía proporciona el procedimiento completo, paso a paso y 100% ejecutable para desplegar el bot de trading **Tradi** en una máquina virtual de Google Cloud Platform bajo el nivel **Always Free** (costo $0.00 USD recurrente), ejecutándose como un demonio continuo 24/7 con `systemd`.

---

## Tabla de Contenidos

1. [Creación de Cuenta en Google Cloud Platform y Proyecto](#sección-1-creación-de-cuenta-en-google-cloud-platform-y-proyecto)
2. [Habilitación de Compute Engine API](#sección-2-habilitación-de-compute-engine-api)
3. [Creación de la VM e2-micro (Always Free)](#sección-3-creación-de-la-vm-e2-micro-always-free)
4. [Configuración de Reglas de Firewall](#sección-4-configuración-de-reglas-de-firewall)
5. [Conexión mediante SSH a la Instancia](#sección-5-conexión-mediante-ssh-a-la-instancia)
6. [Aprovisionamiento Automatizado con setup_gcp.sh](#sección-6-aprovisionamiento-automatizado-con-setup_gcpsh)
7. [Verificación del Servicio Systemd y Monitoreo de Logs](#sección-7-verificación-del-servicio-systemd-y-monitoreo-de-logs)
8. [Obtención de la API Key Gratuita de Gemini en Google AI Studio](#sección-8-obtención-de-la-api-key-gratuita-de-gemini-en-google-ai-studio)
9. [Verificación de Señales e Interacción 1-Clic en Telegram](#sección-9-verificación-de-señales-e-interacción-1-clic-en-telegram)

---

## Sección 1: Creación de Cuenta en Google Cloud Platform y Proyecto

Google Cloud Platform ofrece un nivel gratuito permanente (**Always Free Tier**) que incluye una instancia de máquina virtual `e2-micro` mensual sin costo en regiones seleccionadas de EE.UU. (`us-central1`, `us-east1` o `us-west1`), además de un crédito de prueba de $300 USD durante los primeros 90 días.

### Opción A: Vía Consola Web (Navegador)
1. Accede a [Google Cloud Console](https://console.cloud.google.com/).
2. Inicia sesión con tu cuenta de Google y acepta los términos del servicio.
3. En la barra superior, haz clic en el selector de proyectos y luego en **"Proyecto Nuevo"** (New Project).
4. Asigna como nombre del proyecto: `tradi-bot-prod`.
5. Haz clic en **"Crear"** y asegúrate de seleccionarlo como proyecto activo en la barra superior.

### Opción B: Vía Google Cloud SDK (gcloud CLI)
Si utilizas la terminal local o Google Cloud Shell, ejecuta los siguientes comandos directos:

```bash
# Iniciar sesión en GCP
gcloud auth login

# Crear el proyecto del bot
gcloud projects create tradi-bot-prod --name="Tradi Bot"

# Establecerlo como proyecto predeterminado
gcloud config set project tradi-bot-prod
```

---

## Sección 2: Habilitación de Compute Engine API

Antes de crear máquinas virtuales en GCP, es indispensable activar la API de Compute Engine.

### Opción A: Vía Consola Web
1. En el menú de navegación lateral (☰), dirígete a **Compute Engine** > **Instancias de VM**.
2. Espera a que la consola inicialice y haz clic en el botón azul **"Habilitar"** (Enable API). El proceso puede demorar entre 1 y 2 minutos.

### Opción B: Vía gcloud CLI
Ejecuta el siguiente comando para habilitar la API de forma directa:

```bash
gcloud services enable compute.googleapis.com --project=tradi-bot-prod
```

---

## Sección 3: Creación de la VM e2-micro (Always Free)

Para garantizar que la máquina virtual no genere cargos mensuales, debe cumplir con los parámetros exactos del nivel Always Free de GCP:
- **Tipo de máquina:** `e2-micro` (2 vCPUs compartidas, 1 GB de memoria RAM).
- **Región:** `us-central1` (Iowa, zona `us-central1-a`).
- **Disco de arranque:** Disco estándar persistente (`pd-standard`) de hasta 30 GB (usaremos 20 GB).
- **Sistema Operativo:** Ubuntu 22.04 LTS (Jammy Jellyfish).

### Opción A: Creación en 1 Línea con gcloud CLI (Recomendado)
Copia y pega este comando en tu terminal para aprovisionar la máquina virtual con todos los parámetros óptimos configurados:

```bash
gcloud compute instances create tradi-vm \
    --project=tradi-bot-prod \
    --zone=us-central1-a \
    --machine-type=e2-micro \
    --network-interface=network-tier=STANDARD,subnet=default \
    --maintenance-policy=MIGRATE \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB \
    --boot-disk-type=pd-standard \
    --boot-disk-device-name=tradi-vm \
    --tags=http-server,https-server,tradi-health
```

### Opción B: Creación manual en Consola Web
1. Ve a **Compute Engine** > **Instancias de VM** y haz clic en **"Crear instancia"**.
2. **Nombre:** `tradi-vm`.
3. **Región:** `us-central1 (Iowa)` | **Zona:** `us-central1-a`.
4. **Configuración de máquina:** Familia `E2`, Tipo de máquina `e2-micro (2 vCPU, 1 GB de memoria)`.
5. **Disco de arranque:** Haz clic en "Cambiar":
   - Sistema operativo: `Ubuntu`.
   - Versión: `Ubuntu 22.04 LTS`.
   - Tipo de disco: `Disco persistente estándar`.
   - Tamaño: `20 GB`.
   - Haz clic en "Seleccionar".
6. **Firewall:** Marca las casillas **"Permitir tráfico HTTP"** y **"Permitir tráfico HTTPS"**.
7. **Opciones avanzadas > Red:** En "Etiquetas de red", escribe `tradi-health`.
8. Haz clic en el botón inferior **"Crear"**.

---

## Sección 4: Configuración de Reglas de Firewall

El bot de Tradi expone un servicio web liviano con endpoints de diagnóstico (`/` y `/health`) en el puerto `8080` para verificación de estado y monitoreo.

### Creación de la Regla de Firewall con gcloud CLI
Ejecuta el siguiente comando para autorizar la entrada al puerto 8080:

```bash
gcloud compute firewall-rules create allow-tradi-health \
    --project=tradi-bot-prod \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:8080,tcp:80,tcp:443 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=tradi-health,http-server
```

### Configuración en Consola Web
1. En el menú lateral, dirígete a **Red de VPC** > **Reglas de firewall**.
2. Haz clic en **"Crear regla de firewall"**.
3. **Nombre:** `allow-tradi-health`.
4. **Red:** `default`.
5. **Dirección del tráfico:** Entrada (Ingress) | **Acción:** Permitir.
6. **Destinos:** Etiquetas de destino especificadas (`tradi-health`).
7. **Filtro de fuente:** Rangos IPv4: `0.0.0.0/0`.
8. **Protocolos y puertos:** Marca "Protocolos y puertos especificados" > `tcp: 8080, 80, 443`.
9. Haz clic en **"Crear"**.

---

## Sección 5: Conexión mediante SSH a la Instancia

Una vez que la instancia esté en estado "En ejecución" (círculo verde), conéctate a la terminal Linux de la máquina virtual.

### Opción A: Botón SSH en la Consola Web
1. En **Compute Engine** > **Instancias de VM**, localiza la fila `tradi-vm`.
2. Haz clic en el botón **"SSH"** ubicado en la columna "Conectar".
3. Se abrirá una ventana de terminal en el navegador web lista para usar.

### Opción B: Mediante gcloud CLI
Desde tu computadora local con Google Cloud SDK:

```bash
gcloud compute ssh tradi-vm --zone=us-central1-a --project=tradi-bot-prod
```

---

## Sección 6: Aprovisionamiento Automatizado con setup_gcp.sh

El repositorio incluye el script de aprovisionamiento desatendido `deploy/setup_gcp.sh`, el cual realiza todas las tareas de configuración en un único comando:
1. Configura un archivo Swap de 2GB (`/swapfile`) para evitar caídas por falta de memoria (OOM) en la VM `e2-micro`.
2. Actualiza los paquetes del sistema e instala librerías de compilación.
3. Instala Python 3.11 nativo vía deadsnakes PPA.
4. Clona el repositorio oficial de Tradi en `/home/ubuntu/tradi`.
5. Crea el entorno virtual (`venv`) e instala `requirements.txt`.
6. Solicita interactivamente tus credenciales para generar `/home/ubuntu/tradi/.env` protegido con permisos `600`.
7. Registra, habilita y arranca el servicio `systemd` para ejecución 24/7.

### Pasos de Ejecución en la VM
Dentro de la consola SSH de la máquina virtual, copia y pega el siguiente bloque:

```bash
# 1. Asegurar herramientas iniciales y clonar repositorio
sudo apt-get update && sudo apt-get install -y git
sudo mkdir -p /home/ubuntu
cd /home/ubuntu
sudo git clone https://github.com/Amziisrae23/tradi.git
cd /home/ubuntu/tradi

# 2. Conceder permisos de ejecución al script
sudo chmod +x deploy/setup_gcp.sh

# 3. Ejecutar el asistente de instalación
./deploy/setup_gcp.sh
```

### Respuestas a las Preguntas Interactivas del Script
Durante la ejecución, el script solicitará tus credenciales:
```text
====================================================================
          CONFIGURACIÓN DE VARIABLES DE ENTORNO (.env)              
====================================================================
Por favor ingrese las credenciales del sistema (presione ENTER para dejar vacías las opcionales):

1. BITUNIX_API_KEY: [Tu API Key de Bitunix o ENTER para modo simulación]
2. BITUNIX_API_SECRET: [Tu Secret Key de Bitunix o ENTER para modo simulación]
3. TELEGRAM_BOT_TOKEN: [Token de tu bot obtenido en @BotFather]
4. TELEGRAM_CHAT_ID: [Tu ID numérico obtenido en @userinfobot]
5. GEMINI_API_KEY: [Tu API Key gratuita obtenida en Google AI Studio (ver Sección 8)]
```

Al finalizar las preguntas, el script arrancará el bot de forma automática.

---

## Sección 7: Verificación del Servicio Systemd y Monitoreo de Logs

El bot corre como un demonio del sistema gestionado por `systemd`, lo que asegura que si la máquina virtual se reinicia o el proceso falla, se reanudará automáticamente en 10 segundos.

### Comprobar Estado del Servicio
```bash
sudo systemctl status tradi.service
```
*Salida esperada:*
```text
● tradi.service - Tradi Quantitative Crypto Trading Copilot 24/7 Daemon
     Loaded: loaded (/etc/systemd/system/tradi.service; enabled; vendor preset: enabled)
     Active: active (running) since ...
```

### Visualizar Logs en Tiempo Real (Seguimiento Continuo)
```bash
sudo journalctl -u tradi.service -f
```
Para salir de la vista de logs en tiempo real, presiona `Ctrl + C`.

### Ver las Últimas 100 Líneas de Registro
```bash
sudo journalctl -u tradi.service -n 100 --no-pager
```

### Probar el Endpoint de Salud (Health Check)
```bash
# Consulta local desde la VM
curl http://localhost:8080/health
```
*Respuesta esperada:*
```json
{"status":"healthy","uptime_hours":0.05,"active_signals":1,"timestamp":"..."}
```

### Comandos de Ciclo de Vida del Bot
```bash
# Reiniciar el bot (tras modificar configuración o .env)
sudo systemctl restart tradi.service

# Detener el bot
sudo systemctl stop tradi.service

# Iniciar el bot
sudo systemctl start tradi.service
```

---

## Sección 8: Obtención de la API Key Gratuita de Gemini en Google AI Studio

Tradi utiliza el modelo **Gemini 2.0 Flash** como Analista Cuantitativo Senior para evaluar la microestructura de mercado, validar confluencias técnicas (Order Blocks, Fair Value Gaps, Sweeps) y justificar cada señal.

Google AI Studio ofrece acceso **100% gratuito** a Gemini 2.0 Flash con las siguientes cuotas de uso:
- **15 Solicitudes por Minuto (RPM)**.
- **1,500 Solicitudes por Día (RPD)**.
- **1,000,000 Tokens por Minuto (TPM)**.
- **Costo mensual:** $0.00 USD.

### Procedimiento para Obtener tu API Key
1. Abre tu navegador e ingresa a [Google AI Studio](https://aistudio.google.com/).
2. Inicia sesión con la misma cuenta de Google utilizada en GCP.
3. En la barra lateral izquierda o menú superior, haz clic en **"Get API key"**.
4. Haz clic en el botón azul **"Create API key"**.
5. En la ventana emergente, selecciona tu proyecto existente de GCP (`tradi-bot-prod`) o haz clic en "Create API key in new project".
6. Copia la clave generada (tiene el formato `AIzaSy...`).

### Actualizar o Modificar la Clave en la VM de GCP
Si dejaste el campo vacío durante la instalación o deseas actualizar la clave más adelante:

```bash
# Editar el archivo .env de forma segura
sudo nano /home/ubuntu/tradi/.env
```
Busca la línea:
```text
GEMINI_API_KEY=tu_api_key_aqui
```
Reemplázala con tu clave real, guarda los cambios con `Ctrl + O` seguido de `ENTER`, y sal con `Ctrl + X`.

Aplica los cambios reiniciando el servicio:
```bash
sudo systemctl restart tradi.service
```

---

## Sección 9: Verificación de Señales e Interacción 1-Clic en Telegram

Una vez en marcha, Tradi iniciará el ciclo de escaneo continuo en Bitunix sobre los 10 pares principales: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `XRPUSDT`, `DOGEUSDT`, `SUIUSDT`, `ADAUSDT`, `AVAXUSDT`, `LINKUSDT` y `BNBUSDT`.

### Mensaje de Inicialización
Al arrancar, el bot enviará un mensaje de bienvenida a tu chat privado de Telegram confirmando:
- Modo de operación (Web Service en puerto 8080).
- Capital inicial configurado ($500 USD, riesgo 2%).
- Estado del motor de IA (Gemini 2.0 Flash activo o degradación elegante).

### Formato de Señal Enriquecida con Análisis IA
Cuando el motor SMC y el modelo de Machine Learning detectan una confluencia de alta probabilidad (Win Rate > 75%), Telegram recibe la señal con el bloque analítico de Gemini antepuesto:

```text
🤖 ANÁLISIS IA — BTCUSDT
━━━━━━━━━━━━━━━━━━━━━
"Estructura alcista confirmada tras barrido de liquidez en soporte clave de 15M y posterior mitigación de Order Block institucional. El ratio R:R de 3.0 hacia TP2 ofrece un perfil asimétrico altamente favorable con probabilidad ML superior al 85%. Se proyecta continuación hacia zona de distribución superior."

🚀 SEÑAL DE TRADING — BTCUSDT
━━━━━━━━━━━━━━━━━━━━━
Dirección: LONG 🟢
Precio Entrada: $64,250.00
Stop Loss: $63,800.00 (-0.70%)
Take Profit 1: $64,925.00 (+1.05% | R:R 1.5)
Take Profit 2: $65,600.00 (+2.10% | R:R 3.0)
Take Profit 3: $66,950.00 (+4.20% | R:R 6.0)
Confianza ML: 88.4% | Razones: Rebote en OB Alcista, FVG 15M Mitigado
```
*Adjunto al mensaje se incluye el gráfico técnico renderizado en alta resolución con las zonas de Order Block y Fair Value Gap resaltadas.*

### Botones Interactivos 1-Clic
Debajo de cada señal encontrarás dos botones interactivos:
- `[ ⚡ EJECUTAR ]`: Envía la orden a Bitunix con cálculo dinámico de posición por criterio Kelly (riesgo $10 USD) y stop loss automático.
- `[ 🗑️ DESCARTAR ]`: Registra el descarte en el historial del bot sin abrir posición.

### Latido de Salud (Heartbeat)
Cada 6 horas de inactividad de mercado, el bot emitirá un reporte de estado en Telegram confirmando que el loop de escaneo se encuentra plenamente operativo en Google Cloud.

---

## Solución de Problemas Frecuentes (Troubleshooting)

### 1. "No recibo mensajes en Telegram"
- **Causa más habitual:** Telegram prohíbe que los bots inicien conversaciones con usuarios por prevención de spam.
- **Solución:** Abre la aplicación de Telegram, busca tu bot por su username y presiona el botón **"Iniciar"** o envía el comando `/start`. Tras esto, reinicia el servicio con `sudo systemctl restart tradi.service`.

### 2. "Permisos denegados al leer .env"
- Verifica que el archivo pertenezca al usuario `ubuntu` con permisos restrictivos `600`:
  ```bash
  ls -la /home/ubuntu/tradi/.env
  # Debe mostrar: -rw------- 1 ubuntu ubuntu ...
  sudo chown ubuntu:ubuntu /home/ubuntu/tradi/.env
  sudo chmod 600 /home/ubuntu/tradi/.env
  ```

### 3. "La memoria RAM parece llena en la VM e2-micro"
- Comprueba que el archivo swap de 2GB esté activo:
  ```bash
  free -h
  swapon --show
  ```
- Si la línea `Swap` muestra `2.0Gi`, el sistema está protegido y el kernel utilizará la memoria virtual para amortiguar los picos de procesamiento de gráficos o inferencia sin agotar la memoria física.

### 4. "Ver error detallado si el servicio falla al iniciar"
- Ejecuta:
  ```bash
  sudo journalctl -u tradi.service -n 50 --no-pager
  ```
  Esto mostrará el traceback exacto de Python para diagnosticar variables faltantes o problemas de conectividad de red.
