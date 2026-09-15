# Reglas de Flujo de Trabajo y Sincronización Multi-Equipo

Este proyecto se trabaja entre dos computadoras (Laptop Trabajo y Laptop Casa) utilizando GitHub y Antigravity.

## 1. Regla de Cierre de Sesión ("terminamos")
Cada vez que el usuario indique que la sesión ha finalizado (e.g., diciendo "terminamos", "listo por hoy", "fin de sesión", "ya me voy", o similar), Antigravity DEBE ejecutar automáticamente el siguiente protocolo:
1. **Analizar los cambios realizados**: Revisar qué archivos se crearon, modificaron o eliminaron en la sesión.
2. **Actualizar `TODO.md`**:
   - Marcar las tareas completadas.
   - Detallar el estado actual del código.
   - Listar con precisión cuáles son las próximas tareas a realizar al abrir el proyecto en la otra computadora.
3. **Guardar cambios en Git**:
   - Ejecutar `git add .`
   - Generar un commit descriptivo y profesional (ejemplo: `feat: implementación de módulo X` o `chore: cierre de sesión y actualización de TODO`).
4. **Subir a GitHub**:
   - Ejecutar `git push` para sincronizar con el repositorio remoto.
5. **Reporte de Entrega (Handover)**:
   - Mostrar un mensaje conciso al usuario confirmando que todo está subido a GitHub y qué es lo primero que se hará en la otra laptop al reanudar.

## 2. Regla de Inicio de Sesión ("continuemos")
Cuando el usuario inicie una nueva sesión de chat o diga "continuemos", "en qué nos quedamos", o "hola":
1. Ejecutar `git pull` para asegurar que se tienen los cambios más recientes de la otra computadora.
2. Leer `TODO.md` y el último commit de Git.
3. Informar al usuario en qué punto exacto se quedó el proyecto y cuál es la tarea inmediata a ejecutar.

## 3. Seguridad de Credenciales y Archivos Locales
- NUNCA agregar archivos `.env`, contraseñas, tokens de API o secretos al control de versiones.
- Mantener siempre actualizado `.env.example` con las variables requeridas (sin valores confidenciales).
