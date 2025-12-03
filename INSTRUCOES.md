# Estadio Interativo - Guia de Execucao (Raspberry Pi, microfone USB)

## Requisitos
- Raspberry Pi com Raspberry Pi OS
- Microfone USB (para nivel de ruido)
- Botao em GPIO17 (BCM) com resistor pull-down (ou ajuste para pull-up no codigo)
- LED em GPIO27 (BCM) com resistor
- Camera Pi v2 (opcional, para deteccao de rostos/emocoes e overlay em video)

## Preparacao do sistema
1) Confirmar que o microfone USB e detetado: `arecord -l` ou use o snippet abaixo (PyAudio).
2) Instalar dependencias:
```bash
sudo apt update
sudo apt install -y python3-gpiozero python3-opencv python3-picamera2 python3-pyaudio  # picamera2 so se quiser usar camera/display
# Se faltar as cascatas do OpenCV (erro de haarcascades), instale:
sudo apt install -y opencv-data
```

## Executar (modo basico, sem video)
```bash
python stadium_interactive.py --interval 0.5
```

## Executar com video/emocoes
```bash
python stadium_interactive.py --display --camera --interval 0.1  --mic-samplerate 44100
# Exemplo com balanço de brancos fixo:
# python stadium_interactive.py --display --camera --interval 0.1 --mic-samplerate 44100 --wb-kelvin 4500
# Exemplo com gains manuais (ajustar tons de pele):
# python stadium_interactive.py --display --camera --interval 0.1 --mic-samplerate 44100 --color-gains 1.8,1.2
# Exemplo combinando Kelvin + gains:
# python stadium_interactive.py --display --camera --interval 0.1 --mic-samplerate 44100 --wb-kelvin 4500 --color-gains 1.4,1.0
```

Argumentos uteis:
- `--mic-device` indice do microfone (None usa o padrao; veja lista pelo pyaudio no snippet abaixo).
- `--mic-samplerate` (padrao 16000) taxa de amostragem de audio (tente 44100 se tiver erro de sample rate).
- `--mic-frames` (padrao 1024) amostras lidas por ciclo (mais alto = leitura mais lenta/pouco ruido).
- `--mic-gain` (padrao 2.0) ganho aplicado ao nivel do micro (baixe se satura rapido).
- `--mic-noise-floor` (padrao 0.02) ruido de fundo a remover (0-0.5). Aumente para ignorar ruido constante.
- `--mic-smoothing` (padrao 0.3) suavizacao exponencial do nivel (0-1). Maior = mais estavel, menos picos.
- `--button-pin` (padrao 17) GPIO do botao (BCM).
- `--led-pin` (padrao 27) GPIO do LED (BCM).
- `--camera` ativa a Camera Pi v2 (necessario para deteccao de rosto/emocao).
- `--display` mostra janela com frame, emocao, ruido, pressao e estado.
- `--resolution` (ex.: `1280x720`) define a resolucao do feed de camera.
- `--wb-kelvin` fixa o balanço de brancos em Kelvin (ex.: 4500). Se omitir, usa AWB automatico.
- `--color-gains` fixa gains R,B (ex.: `1.8,1.2`) e desativa AWB. Pode combinar com `--wb-kelvin`; ambos sao aplicados.

Listar dispositivos de microfone rapidamente:
```bash
python - <<'PY'
import pyaudio
pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    print(i, info.get('name'))
pa.terminate()
PY
```

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
