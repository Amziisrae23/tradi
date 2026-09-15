# Tradi 🚀

Proyecto colaborativo y multiplataforma desarrollado con Antigravity.

---

## 💻 Guía de Trabajo Multi-Equipo (Laptop Trabajo ↔ Laptop Casa)

### 1. Al comenzar a trabajar en cualquier computadora:
Abre el proyecto y dile a Antigravity:
> *"Continuemos"* (o *"¿En qué nos quedamos?"*)
Antigravity ejecutará `git pull`, leerá `TODO.md` y te dirá exactamente el estado del proyecto y qué sigue.

### 2. Al terminar tu jornada de trabajo:
Dile a Antigravity:
> *"Terminamos"* (o *"Listo por hoy"*)
Antigravity actualizará automáticamente `TODO.md`, creará el commit y subirá todo a GitHub con `git push`.

---

## 🛠️ Configuración en tu Laptop de Casa

1. Asegúrate de tener Git instalado.
2. Abre tu terminal y clona el repositorio:
   ```bash
   git clone git@github.com:Amziisrae23/tradi.git
   # O por HTTPS:
   git clone https://github.com/Amziisrae23/tradi.git
   ```
3. Copia el archivo de variables de entorno si aplica:
   ```bash
   cp .env.example .env
   ```
4. Abre Antigravity en la carpeta y escribe: *"Hola, continuemos con el proyecto"*.
