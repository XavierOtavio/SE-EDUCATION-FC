# Testes recomendados (Raspberry Pi, microfone USB)

## Preparacao
- Confirmar que o microfone USB aparece em `arecord -l` ou `python - <<'PY'\nimport pyaudio\npa = pyaudio.PyAudio()\nfor i in range(pa.get_device_count()):\n    print(i, pa.get_device_info_by_index(i).get('name'))\npa.terminate()\nPY`.
- Wiring: botao no GPIO17 (pull-down), LED no GPIO27 com resistor.
- Instalar dependencias: `sudo apt update && sudo apt install -y python3-gpiozero python3-opencv opencv-data python3-picamera2 python3-pyaudio` (picamera2 apenas se usar camera/display).

## Checklist funcional
1. **Arranque (sem video)**: `python stadium_interactive.py --interval 0.5` (sem `--display`/`--camera`). Ver se imprime leituras sem erros.
2. **Ruido baixo**: Ambiente silencioso -> `Ruido` abaixo de ~512, mensagem `Entusiasmo normal`, LED apagado.
3. **Ruido alto + botao pressionado**: Bater palmas/falar alto e pressionar botao -> LED acende e `GOLO`.
4. **Ruido alto + botao solto**: Com ruido alto, soltar botao -> LED apaga e `VAIA`.
5. **Alternancia rapida**: Alternar botao durante ruido alto e confirmar reacao imediata do LED/estado.
6. **Display + emocoes**: `python stadium_interactive.py --display --camera --interval 0.1 --mic-samplerate 44100`. Confirmar:
   - Janela com rostos, bounding boxes e emocao (Feliz se sorriso, Neutro se nao).
   - Overlay com `Ruido`, `Pressao` e `Estado` visivel no frame.
   - LED fisico corresponde ao estado (GOLO/VAIA/Entusiasmo normal).
   - Opcional: testar `--wb-kelvin 4500` para forcar balanço de brancos e ver se as cores estabilizam.
7. **Fecho limpo**: Premir `q` na janela ou `Ctrl+C` no terminal; confirmar que a app fecha sem stack trace e liberta GPIO/camera.

## Diagnostico rapido
- Se erro de dispositivo/entrada no pyaudio: verificar argumento `--mic-device`, `--mic-samplerate` (tentar 44100) e lista de dispositivos.
- Se o botao nunca muda: rever pull-down/pull-up e pino configurado.
- Se o LED fica sempre ON/OFF: testar `--led-pin` correto e rever resistor/ligacao.
- Se a janela nao abre: confirmar `--display` e ambiente grafico; em headless, usar virtual display ou remover `--display`.
