# Estadio Interativo - Guia de Execucao (Raspberry Pi, microfone USB)

## Requisitos
- Raspberry Pi com Raspberry Pi OS
- Microfone USB (para nivel de ruido)
- Botao em GPIO17 (BCM) com resistor pull-down (ou ajuste para pull-up no codigo)
- LED em GPIO27 (BCM) com resistor
- Camera Pi v2 (opcional, para deteccao de rostos/emocoes e overlay em video)

## Preparacao do sistema
1) Confirmar que o microfone USB e detetado: `arecord -l` ou `python - <<'PY'\nimport sounddevice as sd\nprint(sd.query_devices())\nPY`
2) Instalar dependencias:
```bash
sudo apt update
sudo apt install -y python3-gpiozero python3-opencv python3-picamera2 python3-sounddevice  # picamera2 so se quiser usar camera/display
```

## Executar (modo basico, sem video)
```bash
python stadium_interactive.py --interval 0.5
```

## Executar com video/emocoes
```bash
python stadium_interactive.py --display --camera --interval 0.1
```

Argumentos uteis:
- `--mic-device` indice do microfone (None usa o padrao; veja sd.query_devices()).
- `--mic-samplerate` (padrao 16000) taxa de amostragem de audio.
- `--mic-frames` (padrao 1024) amostras lidas por ciclo (mais alto = leitura mais lenta/pouco ruido).
- `--button-pin` (padrao 17) GPIO do botao (BCM).
- `--led-pin` (padrao 27) GPIO do LED (BCM).
- `--camera` ativa a Camera Pi v2 (necessario para deteccao de rosto/emocao).
- `--display` mostra janela com frame, emocao, ruido, pressao e estado.
- `--resolution` (ex.: `1280x720`) define a resolucao do feed de camera.

## O que vera
- Sem `--display`: linhas no terminal, ex.:
  ```
  Ruido= 850 | Pressao=True  | LED=ON  | GOLO
  Ruido= 900 | Pressao=False | LED=OFF | VAIA
  Ruido= 200 | Pressao=False | LED=OFF | Entusiasmo normal
  ```
- Com `--display --camera`: janela de video com:
  - Rostos com bounding boxes e emocao (Feliz se sorriso detetado, Neutro caso contrario).
  - Texto no topo com `Ruido`, `Pressao` e `Estado (GOLO/VAIA/Entusiasmo normal)`.
  - LED fisico segue o mesmo estado.
  - Premir `q` para fechar a janela.

## Criterios de aceitacao cobertos
- Leitura real do nivel de ruido (0-1023) via microfone USB e do botao (True/False).
- Logica:
  - Ruido > 512 e botao pressionado: LED ON e mensagem `GOLO`.
  - Ruido > 512 e botao nao pressionado: LED OFF e mensagem `VAIA`.
  - Ruido <= 512: LED OFF e mensagem `Entusiasmo normal`.
- Com `--display --camera`, a janela mostra rostos com emocao e overlay de ruido/pressao/estado em tempo real; LED reflete o estado.
