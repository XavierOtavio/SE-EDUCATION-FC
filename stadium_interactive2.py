import argparse
import time
import sys
import select
import contextlib
try:
    import termios
    import tty
except ImportError:  # pragma: no cover
    termios = None
    tty = None
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import asyncio
import cv2
import numpy as np
import pyaudio
from gpiozero import LED
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover
    Picamera2 = None
try:
    from bleak import BleakClient
except ImportError:  # pragma: no cover
    BleakClient = None

NOISE_MIN = 0
NOISE_MAX = 1023
THRESHOLD = 100
WINDOW_NAME = "Estadio Interativo"
TEAM_BOX_SIZE = 60
TEAM_BOX_MARGIN = 10
DEFAULT_BLE_UUID = "0000fff3-0000-1000-8000-00805f9b34fb"

# Display mode: 'debug' | 'normal' | 'demo' (demo mutes debug logs and simplifies overlays)
DISPLAY_MODE = "normal"


def bgr_to_primary_label_and_bgr(bgr: Tuple[int, int, int]) -> Tuple[str, Tuple[int, int, int]]:
    """Map BGR color to primary (red/green/blue) label and saturated BGR tuple.

    Returns (label, saturated_bgr)
    """
    b, g, r = bgr
    # Prefer the channel with maximum value. In case of ties, prefer red.
    if r >= g and r >= b:
        return "red", (0, 0, 255)
    if g >= r and g >= b:
        return "green", (0, 255, 0)
    return "blue", (255, 0, 0)


@dataclass
class FaceEmotion:
    x: int
    y: int
    w: int
    h: int
    emotion: str


class EmotionDetector:
    """
    Deteta faces e estima emocao simples:
    - Feliz: sorriso detetado
    - Triste: olhos detetados mas sem sorriso
    - Zangado: sem olhos/sorriso (heuristica para rostos com sobrancelhas cerradas/olhos menos visiveis)
    """

    def __init__(self) -> None:
        cascades = self._cascade_dir()
        self.face_cascade = cv2.CascadeClassifier(
            str(cascades / "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(str(cascades / "haarcascade_eye.xml"))
        self.smile_cascade = cv2.CascadeClassifier(
            str(cascades / "haarcascade_smile.xml")
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if self.face_cascade.empty() or self.smile_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("Nao foi possivel carregar cascades do OpenCV.")

    def _cascade_dir(self) -> Path:
        """Resolve o diretorio das cascatas, mesmo quando cv2.data.haarcascades nao existe."""
        candidates: List[Path] = []
        try:
            candidates.append(Path(cv2.data.haarcascades))
        except Exception:
            pass
        try:
            base = Path(cv2.__file__).resolve().parent
            candidates.append(base / "data" / "haarcascades")
        except Exception:
            pass
        candidates.extend(
            [
                Path("/usr/share/opencv4/haarcascades"),
                Path("/usr/local/share/opencv4/haarcascades"),
                Path("/usr/share/opencv/haarcascades"),
                Path("/usr/local/share/opencv/haarcascades"),
            ]
        )
        for c in candidates:
            if (c / "haarcascade_frontalface_default.xml").exists():
                return c
        raise RuntimeError(
            "Nao encontrei cascatas do OpenCV. Instale python3-opencv/opencv-data ou indique o caminho."
        )

    def detect(self, frame_bgr) -> List[FaceEmotion]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
        )
        results: List[FaceEmotion] = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y : y + h, x : x + w]
            smiles = self.smile_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.5, minNeighbors=15
            )
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=6
            )
            if roi_gray.size > 0:
                mean_intensity = float(cv2.mean(roi_gray)[0])
            else:
                mean_intensity = 128.0
            emotion = self._classify(smiles, eyes, mean_intensity)
            results.append(FaceEmotion(x, y, w, h, emotion))
        return results

    def _classify(self, smiles, eyes, mean_intensity: float) -> str:
        has_smile = len(smiles) > 0
        has_eyes = len(eyes) > 0
        if has_smile:
            return "Feliz"
        # Se olhos nao aparecem ou imagem escura, considerar Triste
        if not has_eyes or mean_intensity < 90:
            return "Triste"
        # Caso contrario, consideramos neutro
        return "Neutro"


def dominant_color(frame_bgr, faces: List[FaceEmotion], k: int = 3) -> Tuple[int, int, int]:
    """
    Estima a cor predominante das camisolas: usa regi  o de tronco abaixo da face
    mais larga. Se n o houver faces, usa o frame inteiro.
    """
    try:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Frame vazio")
        h, w, _ = frame_bgr.shape
        if faces:
            # Escolher face mais larga e inferir tronco abaixo.
            face = max(faces, key=lambda f: f.w)
            x1 = max(0, int(face.x - 0.2 * face.w))
            x2 = min(w, int(face.x + 1.2 * face.w))
            y1 = min(h, int(face.y + face.h))  # abaixo do queixo
            y2 = min(h, int(face.y + 2.5 * face.h))  # estender pelo tronco
            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                roi = frame_bgr
        else:
            roi = frame_bgr
        small = cv2.resize(roi, (80, 80))
        data = small.reshape((-1, 3)).astype(np.float32)
        if data.shape[0] < k:
            raise ValueError("Dados insuficientes para k-means")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        counts = np.bincount(labels.flatten())
        dominant = centers[np.argmax(counts)]
        return tuple(int(c) for c in dominant)
    except Exception:
        return (0, 0, 255)  # fallback vermelho


class AudioLevelReader:
    """Le nivel de audio de um microfone USB e devolve escala 0-1023."""

    def __init__(
        self,
        device_index: Optional[int],
        samplerate: int,
        frames: int,
        gain: float,
        noise_floor: float,
        smoothing: float,
        auto_gain: bool,
        target_level: float,
        peak_decay: float,
        max_gain: float,
    ) -> None:
        self.frames = frames
        self.samplerate = samplerate
        self.gain = gain
        self.noise_floor = max(0.0, min(0.5, noise_floor))
        self.smoothing = max(0.0, min(1.0, smoothing))
        self.auto_gain = auto_gain
        self.target_level = max(0.1, min(1.0, target_level))
        self.peak_decay = max(0.0, min(1.0, peak_decay))
        self.max_gain = max(0.1, max_gain)
        self.rolling_peak = 0.1
        self.prev_level: Optional[int] = None
        self.pa = pyaudio.PyAudio()
        self.stream = self._open_stream(device_index, samplerate, frames)
        self.max_int = float(np.iinfo(np.int16).max)

    def _open_stream(
        self, device_index: Optional[int], samplerate: int, frames: int
    ) -> pyaudio.Stream:
        """
        Tenta abrir stream com o sample rate pedido; se falhar, tenta taxas comuns
        (48000, 44100, 16000) e a taxa default do dispositivo.
        """
        candidates = [samplerate, 48000, 44100, 16000]
        try:
            dev_info = (
                self.pa.get_device_info_by_index(device_index)
                if device_index is not None
                else self.pa.get_default_input_device_info()
            )
            default_rate = int(dev_info.get("defaultSampleRate", samplerate))
            if default_rate not in candidates:
                candidates.append(default_rate)
        except Exception:
            pass

        last_err: Optional[Exception] = None
        for rate in candidates:
            try:
                stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=frames,
                    input_device_index=device_index,
                )
                # Guardar a taxa efetivamente usada (pode divergir do pedido inicial)
                self.samplerate = rate
                return stream
            except Exception as err:  # pragma: no cover - depende de hardware
                last_err = err
                continue
        raise RuntimeError(
            f"Nao foi possivel abrir microfone com taxas {candidates}. Ultimo erro: {last_err}"
        )

    def read_level(self) -> Tuple[int, float]:
        data = self.stream.read(self.frames, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16)
        if samples.size == 0:
            return 0, 0.0
        samples_f = samples.astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(samples_f))))
        peak = float(np.max(np.abs(samples_f)))
        rms_norm = rms / self.max_int
        peak_norm = peak / self.max_int
        if rms_norm > self.noise_floor:
            rms_norm = (rms_norm - self.noise_floor) / max(1e-6, (1 - self.noise_floor))
        else:
            rms_norm = 0.0
        # Usa o maior dos dois para evitar que sons fortes pareçam fracos.
        level_norm = max(rms_norm, peak_norm)
        if self.auto_gain:
            self.rolling_peak = max(rms_norm, self.rolling_peak * self.peak_decay)
            dyn_gain = self.gain
            if self.rolling_peak > 1e-6:
                dyn_gain *= self.target_level / self.rolling_peak
            dyn_gain = min(dyn_gain, self.max_gain)
        else:
            dyn_gain = self.gain
        level_raw = int(min(NOISE_MAX, level_norm * dyn_gain * NOISE_MAX))
        if self.smoothing <= 0:
            level_smooth = level_raw
        else:
            if self.prev_level is None:
                level_smooth = level_raw
            else:
                alpha = self.smoothing
                level_smooth = int(self.prev_level + alpha * (level_raw - self.prev_level))
        self.prev_level = level_smooth
        # FFT simples para pico de frequencia
        samples_float = samples.astype(np.float32)
        window = np.hanning(len(samples_float))
        spectrum = np.fft.rfft(samples_float * window)
        freqs = np.fft.rfftfreq(len(samples_float), d=1.0 / self.samplerate)
        peak_idx = int(np.argmax(np.abs(spectrum)))
        peak_freq = float(freqs[peak_idx]) if peak_idx < len(freqs) else 0.0
        return level_smooth, peak_freq

    def close(self) -> None:
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.pa.terminate()


class DummyAudioLevelReader:
    """Substituto quando o microfone nao deve ser usado."""

    def close(self) -> None:
        pass

    def read_level(self) -> Tuple[int, float]:
        return 0, 0.0


class BLEController:
    """Envia cor via BLE. Cada envio cria e fecha a ligacao para evitar conflitos de loop asyncio."""

    def __init__(self, device_mac: str, write_uuid: str, enabled: bool) -> None:
        self.device_mac = device_mac
        self.write_uuid = write_uuid
        self.enabled = enabled and BleakClient is not None
        self.last_color: Optional[Tuple[int, int, int]] = None
        if enabled and BleakClient is None:
            if DISPLAY_MODE == "debug":
                print("[BLE] bleak nao instalado; desative BLE ou instale bleak.")

    async def _send_once(self, r: int, g: int, b: int) -> None:
        """Liga, envia e desliga no mesmo loop para evitar conflitos de loops."""
        if not self.enabled:
            return
        try:
            async with BleakClient(self.device_mac) as client:
                if not client.is_connected:
                    await client.connect()
                if not client.is_connected:
                    if DISPLAY_MODE == "debug":
                        print("[BLE] Falha ao ligar. Verifique o controlador.")
                    return
                cmd = bytes([0x7E, 0x00, 0x05, 0x03, r, g, b, 0x00, 0xEF])
                await client.write_gatt_char(self.write_uuid, cmd)
                if DISPLAY_MODE == "debug":
                    print(f"[BLE] Cor enviada (R={r} G={g} B={b}).")
        except Exception as exc:
            if DISPLAY_MODE == "debug":
                print(f"[BLE] Erro ao enviar cor: {exc}")

    def send_color(self, bgr: Tuple[int, int, int]) -> None:
        """Enviar cor apenas se for diferente da anterior, com retry."""
        if not self.enabled:
            return
        bgr_primary = self._to_primary(bgr)
        if bgr_primary == self.last_color:
            return  # Nao enviar se for a mesma cor
        self.last_color = bgr_primary
        b, g, r = bgr_primary
        for attempt in range(2):  # duas tentativas para cobrir falha inicial
            try:
                asyncio.run(self._send_once(r, g, b))
                return
            except RuntimeError as exc:
                if DISPLAY_MODE == "debug":
                    print(f"[BLE] Erro ao correr loop asyncio (tentativa {attempt+1}): {exc}")
                time.sleep(0.2)
        if DISPLAY_MODE == "debug":
            print("[BLE] Nao foi possivel enviar cor apos retries.")

    def turn_off_leds(self) -> None:
        """Desligar LEDs (enviar preto)"""
        self.send_color((0, 0, 0))

    def close(self) -> None:
        """Sem estado persistente para fechar quando usamos liga/desliga por envio."""
        pass

    def _to_primary(self, bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Satura a cor para o canal dominante (B, G ou R) para ficar 255/0/0 etc.
        Preserva branco (ou quase branco/cinza claro) e preto quando aplicável.
        """
        b, g, r = bgr
        max_val = max(b, g, r)
        min_val = min(b, g, r)
        # Preto / muito escuro
        if max_val < 15:
            return (0, 0, 0)
        # Branco ou quase branco/cinza claro: mantém branco
        if max_val > 200 and (max_val - min_val) < 40:
            return (255, 255, 255)
        # Canal dominante
        if b == max_val and b > g and b > r:
            return (255, 0, 0)  # azul dominante em BGR
        if g == max_val and g > b and g > r:
            return (0, 255, 0)  # verde dominante
        # Se r for o maior (ou empate), assume vermelho dominante
        return (0, 0, 255)

class OptimizedCameraProcessor:
    def __init__(self, resolution=(320, 240), buffer_size=3):
        self.resolution = resolution
        self.frame_buffer = deque(maxlen=buffer_size)
        self.emotion_buffer = deque(maxlen=10)
        self.lock = threading.Lock()
        
        # Carregar cascata Haar otimizada
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        
        # Tentar ativar aceleração de hardware
        self._init_hardware_acceleration()
        
    def _init_hardware_acceleration(self):
        """Ativa aceleração de hardware se disponível"""
        try:
            # Para Raspberry Pi com MMAL
            cv2.setUseOptimized(True)
            cv2.useIPP(True)
        except:
            pass
    
    def capture_frames(self, camera_source=0, fps_target=15):
        """Thread para captura contínua de frames"""
        cap = cv2.VideoCapture(camera_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps_target)
        
        # Usar aceleração de hardware se disponível
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_time = 1.0 / fps_target
        
        while True:
            start = time.time()
            ret, frame = cap.read()
            
            if ret:
                with self.lock:
                    self.frame_buffer.append({
                        'frame': frame.copy(),
                        'timestamp': datetime.now()
                    })
            
            elapsed = time.time() - start
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)
    
    def detect_faces_optimized(self, frame):
        """Detecção de faces com parâmetros otimizados"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Parâmetros mais rigorosos para limitar detecções
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,        # Mais alto = menos detecções
            minNeighbors=5,         # Mais alto = mais rigoroso
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(20, 20),       # Tamanho mínimo de face
            maxSize=(self.resolution[0]//2, self.resolution[1]//2)
        )
        
        return faces
    
    def process_emotions_batch(self, frames_batch):
        """Processar múltiplos frames em batch para emoções"""
        emotions = []
        
        for frame_data in frames_batch:
            frame = frame_data['frame']
            faces = self.detect_faces_optimized(frame)
            
            if len(faces) > 0:
                # Processar apenas a maior face para eficiência
                (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
                face_roi = frame[y:y+h, x:x+w]
                
                # Aqui você processaria com modelo de emoções
                emotion_data = {
                    'face_rect': (x, y, w, h),
                    'timestamp': frame_data['timestamp']
                }
                emotions.append(emotion_data)
        
        return emotions


class AudioProcessor(threading.Thread):
    def __init__(self, device_id=2, sample_rate=44100):
        super().__init__(daemon=True)
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = True
        
    def run(self):
        """Processar áudio continuamente em thread separada"""
        try:
            import sounddevice as sd
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"Audio status: {status}")
                
                try:
                    self.audio_queue.put_nowait(indata.copy())
                except queue.Full:
                    pass  # Descartar frame se fila está cheia
            
            with sd.InputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=2048
            ):
                while self.running:
                    time.sleep(0.1)
        except ImportError:
            print("sounddevice não instalado")


class LEDController(threading.Thread):
    def __init__(self, ble_device=None, ble_uuid=None, update_interval=0.1):
        super().__init__(daemon=True)
        self.ble_device = ble_device
        self.ble_uuid = ble_uuid
        self.update_interval = update_interval
        self.command_queue = queue.Queue()
        self.running = True
        self.last_update = time.time()
        
    def run(self):
        """Processar comandos LED continuamente"""
        while self.running:
            try:
                command = self.command_queue.get(timeout=self.update_interval)
                self._send_to_ble(command)
                self.last_update = time.time()
            except queue.Empty:
                pass
    
    def _send_to_ble(self, command):
        """Enviar comando via BLE"""
        # Implementar comunicação BLE aqui
        pass
    
    def queue_command(self, command):
        """Adicionar comando à fila"""
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            pass


class ImageDisplayBuffer:
    """Buffer para exibição de imagens, não em tempo real"""
    def __init__(self, buffer_size=5, display_interval=0.5):
        self.buffer = deque(maxlen=buffer_size)
        self.display_interval = display_interval
        self.last_display = time.time()
        self.lock = threading.Lock()
    
    def add_frame(self, frame, metadata=None):
        with self.lock:
            self.buffer.append({
                'frame': frame,
                'metadata': metadata,
                'timestamp': time.time()
            })
    
    def display_if_needed(self, window_name='Stadium Interactive'):
        """Exibir apenas se passou o intervalo"""
        current_time = time.time()
        
        if current_time - self.last_display >= self.display_interval:
            with self.lock:
                if self.buffer:
                    latest = self.buffer[-1]
                    cv2.imshow(window_name, latest['frame'])
                    self.last_display = current_time
                    return True
        
        return False


class PiBackend:
    """
    Le sensores no Raspberry Pi: microfone USB para ruido, botao GPIO e LED, camera opcional.
    """

    def __init__(
        self,
        mic_device: Optional[int],
        mic_samplerate: int,
        mic_frames: int,
        mic_gain: float,
        mic_noise_floor: float,
        mic_smoothing: float,
        mic_auto_gain: bool,
        mic_target_level: float,
        mic_peak_decay: float,
        mic_max_gain: float,
        mic_enabled: bool,
        ble_device: str,
        ble_uuid: str,
        ble_enabled: bool,
        led_pin: int,
        use_camera: bool,
        display: bool,
        resolution: Tuple[int, int],
        wb_kelvin: Optional[int],
        color_gains: Optional[Tuple[float, float]],
    ) -> None:
        if mic_enabled:
            self.audio = AudioLevelReader(
                mic_device,
                mic_samplerate,
                mic_frames,
                mic_gain,
                mic_noise_floor,
                mic_smoothing,
                mic_auto_gain,
                mic_target_level,
                mic_peak_decay,
                mic_max_gain,
            )
        else:
            self.audio = DummyAudioLevelReader()
        self.led = LED(led_pin)

        self.detector: Optional[EmotionDetector] = None
        self.camera: Optional[Picamera2] = None
        self.display = display
        self.ble = BLEController(ble_device, ble_uuid, ble_enabled)
        # Controle de blink simples para evitar conflitos de asyncio/overrides
        self._blink_lock = threading.Lock()
        self._is_blinking = False
        
        # Buffer para frames e emoções
        self.frame_buffer = deque(maxlen=3)
        self.frame_lock = threading.Lock()
        self.capture_thread: Optional[threading.Thread] = None
        self.camera_running = False
        
        # Armazenar cor da equipa para usar quando LEDs forem ativados
        self.team_color: Tuple[int, int, int] = (0, 0, 0)  # Vermelho por defeito
        self.team_color_lock = threading.Lock()

        if use_camera or display:
            if Picamera2 is None:
                raise RuntimeError(
                    "picamera2 nao instalado; instale python3-picamera2 ou desative --camera/--display."
                )
            self.camera = Picamera2()
            # Configurar resolução otimizada (320x240) uma única vez
            config = self.camera.create_preview_configuration({
                'format': 'RGB888',
                'size': resolution
            })
            self.camera.configure(config)
            self.camera.start()
            self.detector = EmotionDetector()
            
            # Iniciar thread de captura contínua
            self.camera_running = True
            self.capture_thread = threading.Thread(target=self._capture_frames_continuous, daemon=True)
            self.capture_thread.start()

        time.sleep(0.05)

    def _capture_frames_continuous(self) -> None:
        """Thread para capturar frames continuamente sem bloquear o loop principal"""
        try:
            while self.camera_running and self.camera:
                frame = self.camera.capture_array()
                if frame is not None:
                    with self.frame_lock:
                        self.frame_buffer.append({
                            'frame': cv2.rotate(frame, cv2.ROTATE_180),  # Corrigir orientação
                            'timestamp': datetime.now()
                        })
                time.sleep(0.033)  # ~30 FPS na captura, processamento em 15 FPS
        except Exception as e:
            print(f"[Camera] Erro na captura: {e}")

    def read(self) -> Tuple[int, float]:
        """Ler ruido (0-1023) do microfone USB e pico de frequencia."""
        noise_scaled, peak_freq = self.audio.read_level()
        return noise_scaled, peak_freq

    def set_led(self, on: bool) -> None:
        self.led.on() if on else self.led.off()

    def get_latest_frame(self) -> Optional[dict]:
        """Obter o frame mais recente do buffer sem bloquear"""
        with self.frame_lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]
        return None

    def detect_emotions(self, frame_bgr) -> List[FaceEmotion]:
        if not self.detector or frame_bgr is None:
            return []
        return self.detector.detect(frame_bgr)

    def close(self) -> None:
        self.camera_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.camera:
            self.camera.stop()
        self.led.close()
        self.audio.close()
        self.ble.close()

    def get_team_color(self) -> Tuple[int, int, int]:
        """Obter cor atual da equipa"""
        with self.team_color_lock:
            return self.team_color

    def update_team_color(self, color: Tuple[int, int, int]) -> None:
        """Atualizar cor da equipa detetada"""
        with self.team_color_lock:
            self.team_color = color

    def activate_leds(self) -> None:
        """Ativar LEDs com a cor da equipa"""
        color = self.get_team_color()
        self.ble.send_color(color)
        if DISPLAY_MODE != "demo":
            print(f"[LEDs] Ativados com cor da equipa (RGB={color})")

    def blink_leds(self, color_bgr: Tuple[int, int, int], flashes: int = 3, on_duration: float = 0.1) -> None:
        """Pisca os LEDs num loop simples em thread separada (não bloqueante).
        Usa envios síncronos `send_color` para reduzir problemas com asyncio.
        """
        if not self.ble.enabled:
            return

        def _blink():
            if not self._blink_lock.acquire(blocking=False):
                return  # Já a piscar
            try:
                self._is_blinking = True
                primary = self.ble._to_primary(color_bgr)
                for _ in range(flashes):
                    self.ble.send_color(primary)
                    time.sleep(on_duration)
                    self.ble.send_color((0, 0, 0))
                    time.sleep(on_duration)
                # Restaurar cor padrão (branco)
                self.ble.send_color((255, 255, 255))
            finally:
                self._is_blinking = False
                try:
                    self._blink_lock.release()
                except Exception:
                    pass

        t = threading.Thread(target=_blink, daemon=True)
        t.start()

    def deactivate_leds(self) -> None:
        """Desativar LEDs"""
        self.ble.turn_off_leds()
        if DISPLAY_MODE != "demo":
            print("[LEDs] Desativados")


def decide(noise: int) -> Tuple[str, bool]:
    """Retorna (mensagem, led_on) segundo os criterios apenas de ruido."""
    if noise > THRESHOLD:
        return "Ruido alto", True
    return "Ruido normal", False


def classify_sound(noise_level: int, peak_freq: float, amp_duration: float) -> Tuple[str, str]:
    """
    Usa amplitude, pico de frequencia e tempo de som mantido para distinguir
    gritos (golo) de vaias graves e de conversa normal.
    """
    if DISPLAY_MODE == "debug":
        print(
            f"Peak Frequency: {peak_freq:.1f} Hz, Noise Level: {noise_level}, Duracao: {amp_duration:.2f}s"
        )
    # Limiares calibrados com os valores que indicou:
    # - Golo: freq > 500 Hz e ruido > 600 (ou pico alto prolongado)
    # - Vaia: freq 150-250 Hz e ruido ~300
    # - Conversa: freq 50-500 Hz e ruido 100-300 (neutro)
    if  (amp_duration > 1.0 and peak_freq > 450 and noise_level > 500):
        return "Som: Grito prolongado", "golo"
    if (150 <= peak_freq <= 320 and noise_level >= 300 and amp_duration > 1.0):
        return "Som: Vaia prolongada", "vaia"
    # Se nÇœo atingir os limiares acima, considerar neutro/conversa.
    return "Som: Neutro", "neutro"


def draw_overlay(
    frame,
    faces: List[FaceEmotion],
    noise: int,
    message: str,
    team_color: Tuple[int, int, int],
    sound_label: str,
) -> None:
    """Desenha bounding boxes, emocao e info de ruido/estado no frame."""
    for f in faces:
        cv2.rectangle(frame, (f.x, f.y), (f.x + f.w, f.y + f.h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f.emotion,
            (f.x, max(f.y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    overlay_text = f"Ruido: {noise}  Estado: {message}"
    cv2.putText(
        frame,
        overlay_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        sound_label,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 0),
        2,
        cv2.LINE_AA,
    )
    h, w, _ = frame.shape
    x1 = w - TEAM_BOX_MARGIN - TEAM_BOX_SIZE
    y1 = h - TEAM_BOX_MARGIN - TEAM_BOX_SIZE
    x2 = w - TEAM_BOX_MARGIN
    y2 = h - TEAM_BOX_MARGIN
    cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, thickness=-1)
    cv2.putText(
        frame,
        "Equipa",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


class DisplayManager:
    """Gerencia o que aparece no ecrã de forma centralizada e ajustável por modo.

    - debug: mostra muitos detalhes e logs
    - normal: mostra info útil e alguns bounding boxes
    - demo: minimalista, sem logs de debug e com overlays discretos
    """

    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode
        # Track per-team statistics for red/green/blue: each bucket holds golo, vaia, Feliz, Triste
        self.team_stats = {
            "red": {"golo": 0, "vaia": 0, "Feliz": 0, "Triste": 0},
            "green": {"golo": 0, "vaia": 0, "Feliz": 0, "Triste": 0},
            "blue": {"golo": 0, "vaia": 0, "Feliz": 0, "Triste": 0},
        }
        # load team symbols (optional); keep original size and allow resizing per-frame
        base = Path(__file__).resolve().parent
        self.symbol_files = {
            "red": base / "benfica.png",
            "blue": base / "porto.png",
            "green": base / "sporting.png",
        }
        self.symbols = {}
        for k, p in self.symbol_files.items():
            try:
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is not None and img.size > 0:
                    self.symbols[k] = img
            except Exception:
                self.symbols[k] = None
        self._last_demo_label: Optional[str] = None
        self._active_label: Optional[str] = None

    def _color_to_key(self, color: Tuple[int, int, int]) -> str:
        return f"{color[0]}-{color[1]}-{color[2]}"

    def update_stats(self, faces: List[FaceEmotion], sound_kind: Optional[str], team_color: Tuple[int, int, int]) -> None:
        # Map team color to primary label
        label, _ = bgr_to_primary_label_and_bgr(team_color)

        # update sound counters per team (only golo/vaia)
        if sound_kind in ("golo", "vaia"):
            self.team_stats[label][sound_kind] += 1

        # update emotion counts per team (only count Feliz and Triste)
        for f in faces:
            if f.emotion == "Feliz":
                self.team_stats[label]["Feliz"] += 1
            elif f.emotion == "Triste":
                self.team_stats[label]["Triste"] += 1
    def set_active_team(self, team_label: Optional[str]) -> None:
        self._active_label = team_label

    def render(self, frame, faces: List[FaceEmotion], noise: int, message: str, team_color: Tuple[int, int, int], sound_label: str, sound_kind: str) -> None:
        h, w, _ = frame.shape

        # In demo mode keep overlays minimal: only predominant emotion and tiny team box
        if self.mode == "demo":
            # Predominant emotion
            pred = self._predominant_emotion(faces)
            if pred:
                cv2.putText(frame, f"Emocao: {pred}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Small counters top-right (per-team stats)
            self._draw_team_stats(frame)

            # In demo mode, render central team symbol (if available) instead of corner box
            label, _ = bgr_to_primary_label_and_bgr(team_color)
            if label in self.symbols and self.symbols.get(label) is not None:
                self._render_demo_symbol(frame, label)

            return

        # normal/debug mode: show more info but keep layout tidy
        # Draw small summary at top-left
        pred = self._predominant_emotion(faces)
        if pred:
            cv2.putText(frame, f"Emocao: {pred}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # If debug, draw individual bounding boxes and emotions
        if self.mode == "debug":
            for f in faces:
                cv2.rectangle(frame, (f.x, f.y), (f.x + f.w, f.y + f.h), (0, 255, 0), 2)
                cv2.putText(frame, f.emotion, (f.x, max(f.y - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # team color box: show only in normal/demo (hide in debug)
        if self.mode in ("normal", "demo"):
            sat_color = bgr_to_primary_label_and_bgr(team_color)[1]
            x1 = w - TEAM_BOX_MARGIN - TEAM_BOX_SIZE
            y1 = TEAM_BOX_MARGIN
            x2 = w - TEAM_BOX_MARGIN
            y2 = TEAM_BOX_MARGIN + TEAM_BOX_SIZE
            cv2.rectangle(frame, (x1, y1), (x2, y2), sat_color, thickness=-1)

        # draw per-team statistics according to mode
        self._draw_team_stats(frame)

    def _predominant_emotion(self, faces: List[FaceEmotion]) -> Optional[str]:
        if not faces:
            return None
        counts = {"Feliz": 0, "Triste": 0, "Neutro": 0}
        for f in faces:
            if f.emotion in counts:
                counts[f.emotion] += 1
        return max(counts.items(), key=lambda it: it[1])[0]
    
    def _render_demo_symbol(self, frame, label: str) -> None:
        """Overlay the team symbol (with alpha) centered and scaled to frame size."""
        img = self.symbols.get(label)
        if img is None:
            return
        h, w, _ = frame.shape
        # compute desired symbol size: cap to 40% of width and 40% of height
        max_w = int(w * 0.4)
        max_h = int(h * 0.4)
        ih, iw = img.shape[:2]
        scale = min(max_w / iw, max_h / ih, 1.0)
        nw = max(16, int(iw * scale))
        nh = max(16, int(ih * scale))
        try:
            symbol = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        except Exception:
            symbol = img

        # position center
        cx = w // 2
        cy = h // 2
        sx = max(0, cx - nw // 2)
        sy = max(0, cy - nh // 2)
        ex = min(w, sx + nw)
        ey = min(h, sy + nh)

        # Overlay handling alpha channel
        if symbol.shape[2] == 4:
            alpha = symbol[:, :, 3] / 255.0
            for c in range(3):
                frame[sy:ey, sx:ex, c] = (alpha * symbol[:, :, c] + (1 - alpha) * frame[sy:ey, sx:ex, c]).astype(frame.dtype)
        else:
            frame[sy:ey, sx:ex] = symbol
        # remember last label
        self._last_demo_label = label
    def _draw_team_stats(self, frame):
        # New layout behaviour:
        # - In normal/demo: single large container (avoids top-right team box)
        # - In debug: three equal sections across bottom occupying 1/3 width each
        h, w, _ = frame.shape

        # team order and display names
        order = [("red", (0, 0, 255), "Benfica"), ("green", (0, 255, 0), "Sporting"), ("blue", (255, 0, 0), "Porto")]

        # helper: rounded rect usable in both branches
        def rounded_rect(img, x1, y1, x2, y2, color, radius=6):
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)

        # team square region to avoid (top-right)
        box_w = TEAM_BOX_SIZE
        box_h = TEAM_BOX_SIZE
        box_x1 = w - TEAM_BOX_MARGIN - box_w
        box_y1 = TEAM_BOX_MARGIN

        if self.mode in ("normal", "demo"):
            # Fixed bottom container: occupies a band at the bottom of the frame
            margin = 8
            cont_h = min(120, int(h * 0.32))
            cont_x = margin
            cont_w = w - margin * 2
            cont_y = h - margin - cont_h

            # draw panel background only in demo (normal should be transparent)
            shadow_offset = 3
            panel_bg = (28, 28, 28)
            if self.mode == "demo":
                cv2.rectangle(frame, (cont_x + shadow_offset, cont_y + shadow_offset), (cont_x + cont_w + shadow_offset, cont_y + cont_h + shadow_offset), (15,15,15), -1)
                cv2.rectangle(frame, (cont_x, cont_y), (cont_x + cont_w, cont_y + cont_h), panel_bg, -1)

            # show only active team
            label = self._active_label
            if label not in ("red", "green", "blue"):
                return
            stats = self.team_stats.get(label, {"golo": 0, "vaia": 0, "Feliz": 0, "Triste": 0})

            # compute max for normalization
            max_stat = max(1, stats.get("golo",0), stats.get("vaia",0), stats.get("Feliz",0), stats.get("Triste",0))

            # map label to color
            color_map = {k: c for k, c, _ in order}
            team_color_display = color_map.get(label, (200,200,200))

            # fixed max bar length (not exceed visual area)
            bar_len = min(int(cont_w * 0.62), 360)
            bar_x = cont_x + 14
            bar_h = 10
            gap = 10
            # compute required height for two bars + title
            required_h = 28 + (bar_h + gap) * 2 + 8
            if cont_h < required_h:
                cont_h = required_h
                cont_y = h - margin - cont_h
            # place bars with some top padding
            current_y = cont_y + 28

            # Emotions bar (full words with counts in legend)
            label_text = f"Felizes: {stats['Feliz']}  /  Tristes: {stats['Triste']}"
            cv2.putText(frame, label_text, (bar_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
            b_y1 = current_y + 6
            b_y2 = b_y1 + bar_h
            # shadow and track
            rounded_rect(frame, bar_x + 2, b_y1 + 2, bar_x + bar_len + 2, b_y2 + 2, (10,10,10), radius=6)
            rounded_rect(frame, bar_x, b_y1, bar_x + bar_len, b_y2, (60,60,60), radius=6)
            feliz_len = int(bar_len * (stats.get("Feliz",0) / max_stat))
            triste_len = int(bar_len * (stats.get("Triste",0) / max_stat))
            if feliz_len > 0:
                rounded_rect(frame, bar_x, b_y1, bar_x + feliz_len, b_y2, team_color_display, radius=6)
            if triste_len > 0:
                rounded_rect(frame, bar_x + feliz_len, b_y1, bar_x + feliz_len + triste_len, b_y2, (80,80,80), radius=6)

            # Sounds bar (full words)
            current_y = b_y2 + gap
            cv2.putText(frame, f"Golo: {stats['golo']}  /  Vaia: {stats['vaia']}", (bar_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
            s_y1 = current_y + 6
            s_y2 = s_y1 + bar_h
            rounded_rect(frame, bar_x + 2, s_y1 + 2, bar_x + bar_len + 2, s_y2 + 2, (10,10,10), radius=6)
            rounded_rect(frame, bar_x, s_y1, bar_x + bar_len, s_y2, (60,60,60), radius=6)
            golo_len = int(bar_len * (stats.get("golo",0) / max_stat))
            vaia_len = int(bar_len * (stats.get("vaia",0) / max_stat))
            if golo_len > 0:
                rounded_rect(frame, bar_x, s_y1, bar_x + golo_len, s_y2, (0,180,0), radius=6)
            if vaia_len > 0:
                rounded_rect(frame, bar_x + golo_len, s_y1, bar_x + golo_len + vaia_len, s_y2, (0,0,180), radius=6)

        else:
            # debug mode: three sections across bottom, each 1/3 width
            margin = 6
            section_w = (w - margin * 2) // 3
            bar_len = int(section_w * 0.6)
            bar_h = 10
            y = h - 44
            # compute global max for normalization across teams
            global_max = 1
            for l in ("red","green","blue"):
                s = self.team_stats.get(l, {})
                global_max = max(global_max, s.get('golo',0), s.get('vaia',0), s.get('Feliz',0), s.get('Triste',0))

            for idx, (label, color, team_name) in enumerate(order):
                sx = margin + idx * section_w
                # Header: abbreviated counts only (no team name)
                stats = self.team_stats.get(label, {"golo": 0, "vaia": 0, "Feliz": 0, "Triste": 0})
                cv2.putText(frame, f"F:{stats['Feliz']} T:{stats['Triste']}", (sx + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
                # thin rounded track with shadow
                track_x = sx + 6
                track_y1 = y + 2
                track_y2 = track_y1 + bar_h
                # shadow
                rounded_rect(frame, track_x + 2, track_y1 + 2, track_x + bar_len + 2, track_y2 + 2, (10,10,10), radius=5)
                rounded_rect(frame, track_x, track_y1, track_x + bar_len, track_y2, (60,60,60), radius=5)
                g_len = int(bar_len * (stats.get('golo',0) / global_max))
                v_len = int(bar_len * (stats.get('vaia',0) / global_max))
                if g_len > 0:
                    rounded_rect(frame, track_x, track_y1, track_x + g_len, track_y2, (0,180,0), radius=5)
                if v_len > 0:
                    rounded_rect(frame, track_x + g_len, track_y1, track_x + g_len + v_len, track_y2, (0,0,180), radius=5)
                # show G/V counts below each track for visibility in debug
                cv2.putText(frame, f"G:{stats['golo']} V:{stats['vaia']}", (track_x, track_y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, cv2.LINE_AA)






class _NonBlockingKeyReader:
    """
    Leitura não bloqueante de teclas no terminal (Linux).
    Em Raspberry Pi funciona; se não houver termios/tty, degrada para "sem teclas" fora do modo display.
    """
    def __init__(self) -> None:
        self._enabled = termios is not None and tty is not None and sys.stdin.isatty()
        self._old_settings = None

    def __enter__(self):
        if self._enabled:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._enabled and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def read_key(self) -> Optional[str]:
        if not self._enabled:
            return None
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
        return None


def run_loop(backend: PiBackend, interval: float, display: bool, mode: str = "normal") -> None:
    global DISPLAY_MODE
    DISPLAY_MODE = mode
    display_manager = DisplayManager(mode)

    print("A ler sensores no Raspberry Pi. Ctrl+C para sair.")
    if display:
        print("Teclas: [G] golo/grito (piscar), [V] vaia (fixo 5s), [N] neutro, [Q] sair.")
    else:
        print("Teclas (no terminal): [g] golo/grito, [v] vaia, [n] neutro, [q] sair.")
        print("Nota: no terminal, as teclas funcionam em modo 'cbreak' (sem Enter), se o TTY o permitir.")

    current_ble_color: Optional[Tuple[int, int, int]] = None

    # Estado de override/animações
    override_mode: Optional[str] = None  # "blink" | "steady" | None

    blink_start = 0.0
    blink_period = 0.5
    blink_steps_total = 6  # 3x (aceso 0.5s + apagado 0.5s) = 3 segundos
    blink_color: Tuple[int, int, int] = (255, 255, 255)

    steady_until = 0.0
    steady_color: Tuple[int, int, int] = (255, 255, 255)

    # Medição de duração de som "alto"
    amp_duration = 0.0
    last_ts = time.time()
    amp_active = False
    high_count = 0
    low_count = 0

    display_buffer = ImageDisplayBuffer(buffer_size=3, display_interval=0.5)

    # Garantir branco por defeito ao arrancar
    backend.ble.send_color((255, 255, 255))
    current_ble_color = (255, 255, 255)

    def trigger_golo(now_ts: float, color_bgr: Tuple[int, int, int]) -> None:
        # Executa blink simples em thread separado para reduzir problemas
        backend.blink_leds(color_bgr, flashes=3, on_duration=0.5)

    def trigger_vaia(now_ts: float, color_bgr: Tuple[int, int, int]) -> None:
        nonlocal override_mode, steady_until, steady_color
        override_mode = "steady"
        steady_until = now_ts + 5.0  # Mantenha a cor fixa por 5 segundos
        steady_color = backend.ble._to_primary(color_bgr)

    key_reader = _NonBlockingKeyReader() if not display else None

    try:
        ctx = key_reader if key_reader is not None else contextlib.nullcontext()
        with ctx:
            prev_sound_kind = "neutro"
            while True:
                now = time.time()
                dt = now - last_ts
                last_ts = now

                noise, peak_freq = backend.read()

                # Heurística para duração de amplitude "alta" (para ajudar na classificação)
                high_thresh = 200
                low_thresh = 120
                is_high = noise > high_thresh or peak_freq > 500
                if is_high:
                    high_count += 1
                    low_count = 0
                else:
                    low_count += 1
                    high_count = 0
                if high_count >= 2:
                    amp_active = True
                if low_count >= 2:
                    amp_active = False
                    amp_duration = 0.0
                if amp_active:
                    amp_duration = min(8.0, amp_duration + dt)

                message, led_on = decide(noise)
                backend.set_led(led_on)

                sound_label, sound_kind = classify_sound(noise, peak_freq, amp_duration)

                # Atualizar frame/cor da equipa (se houver display/câmara) ou usar última cor conhecida
                frame_data = backend.get_latest_frame() if display else None
                faces: List[FaceEmotion] = []
                if frame_data is not None:
                    frame_rgb = frame_data["frame"]
                    faces = backend.detect_emotions(frame_rgb)
                    team_color = dominant_color(frame_rgb, faces)
                    backend.update_team_color(team_color)
                    # Debounce sound counting: only increment once when sound transitions to golo/vaia
                    sound_to_count = None
                    if sound_kind in ("golo", "vaia") and prev_sound_kind != sound_kind:
                        sound_to_count = sound_kind
                    display_manager.set_active_team(bgr_to_primary_label_and_bgr(team_color)[0])
                    display_manager.update_stats(faces, sound_to_count, team_color)

                    # Demo mode: show a solid background of the primary team color instead of camera frames
                    if DISPLAY_MODE == "demo":
                        h, w, _ = frame_rgb.shape
                        blank = np.zeros_like(frame_rgb)
                        # fill with saturated primary team color (BGR)
                        _, saturated = bgr_to_primary_label_and_bgr(team_color)
                        blank[:] = saturated
                        display_manager.set_active_team(bgr_to_primary_label_and_bgr(team_color)[0])
                        display_manager.render(blank, faces, noise, message, team_color, sound_label, sound_kind)
                        cv2.imshow(WINDOW_NAME, blank)
                    else:
                        # normal/debug: render the actual camera frame with overlays
                        display_manager.render(frame_rgb, faces, noise, message, team_color, sound_label, sound_kind)
                        display_buffer.add_frame(frame_rgb, {"faces": faces, "sound": sound_label})
                        display_buffer.display_if_needed(WINDOW_NAME)
                else:
                    team_color = backend.get_team_color()
                    if DISPLAY_MODE == "debug":
                        print(f"Ruido={noise:4d} | LED={'ON ' if led_on else 'OFF'} | {message} | {sound_label}")
                    elif DISPLAY_MODE == "normal":
                        print(f"Ruido={noise:4d} | {message} | {sound_label}")
                    # demo mode: show solid team color background even without camera frames
                    if DISPLAY_MODE == "demo":
                        # create a small default frame if we don't have one yet
                        h, w = 480, 640
                        try:
                            # try to infer from backend camera if available
                            latest = backend.get_latest_frame()
                            if latest is not None:
                                h, w, _ = latest["frame"].shape
                        except Exception:
                            pass
                        blank = np.zeros((h, w, 3), dtype=np.uint8)
                        _, saturated = bgr_to_primary_label_and_bgr(team_color)
                        blank[:] = saturated
                        display_manager.set_active_team(bgr_to_primary_label_and_bgr(team_color)[0])
                        display_manager.render(blank, [], noise, message, team_color, sound_label, sound_kind)
                        cv2.imshow(WINDOW_NAME, blank)

                # Teclas (modo display via OpenCV; modo terminal via stdin)
                key: Optional[str] = None
                if display:
                    k = cv2.waitKey(1) & 0xFF
                    if k != 255:
                        key = chr(k)
                else:
                    key = key_reader.read_key() if key_reader is not None else None

                if key in ("q", "Q"):
                    break
                if key in ("g", "G"):
                    trigger_golo(now, team_color)
                if key in ("v", "V"):
                    trigger_vaia(now, team_color)
                if key in ("n", "N"):
                    override_mode = None

                # Triggers automáticos por áudio (não cancelam override em curso; apenas o substituem)
                if sound_kind == "golo":
                    trigger_golo(now, team_color)
                elif sound_kind == "vaia":
                    trigger_vaia(now, team_color)

                # update prev_sound_kind after triggers so we debounce counting
                prev_sound_kind = sound_kind

                # Determinar cor desejada, com prioridade para o override ativo
                desired_color: Tuple[int, int, int] = (255, 255, 255)

                # Blink de golo é tratado por `backend.blink_leds` em thread separada.

                if override_mode == "steady":
                    if now >= steady_until:
                        override_mode = None
                        desired_color = (255, 255, 255)  # Cor branca após o tempo de fixação
                    else:
                        desired_color = steady_color

                # Enviar via BLE apenas se mudou
                if desired_color != current_ble_color:
                    backend.ble.send_color(desired_color)
                    current_ble_color = desired_color

                time.sleep(interval)
    except KeyboardInterrupt:
        print("\nTerminando...")
    finally:
        backend.deactivate_leds()
        backend.close()
        if display:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estadio interativo em Raspberry Pi (microfone USB + GPIO + camera/emocao)."
    )
    parser.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="Indice do dispositivo de microfone (None usa o padrao do SO).",
    )
    parser.add_argument(
        "--mic-samplerate",
        type=int,
        default=16000,
        help="Sample rate para captura de audio.",
    )
    parser.add_argument(
        "--mic-frames",
        type=int,
        default=1024,
        help="Numero de frames lidos por ciclo (quanto maior, mais lento).",
    )
    parser.add_argument(
        "--mic-gain",
        type=float,
        default=2.0,
        help="Ganho aplicado ao audio normalizado (ajuste para evitar saturar).",
    )
    parser.add_argument(
        "--mic-noise-floor",
        type=float,
        default=0.02,
        help="Nivel de ruido de fundo a remover (0-0.5). Aumente se o micro captar ruido constante.",
    )
    parser.add_argument(
        "--mic-smoothing",
        type=float,
        default=0.3,
        help="Fator de suavizacao exponencial (0-1). Valores maiores suavizam mais.",
    )
    parser.add_argument(
        "--mic-auto-gain",
        action="store_true",
        default=False,
        help="Ativa ajuste dinamico de ganho (por defeito fica desligado; use mic-gain manual).",
    )
    parser.add_argument(
        "--mic-target-level",
        type=float,
        default=0.6,
        help="Nivel alvo (0-1) usado no auto-gain para escalar o pico recente.",
    )
    parser.add_argument(
        "--mic-peak-decay",
        type=float,
        default=0.95,
        help="Fator de decaimento do pico (0-1) para o auto-gain.",
    )
    parser.add_argument(
        "--mic-max-gain",
        type=float,
        default=8.0,
        help="Limite superior do ganho dinamico (auto-gain) para evitar saturar.",
    )
    parser.add_argument(
        "--mic-enabled",
        action="store_true",
        default=True,
        help="Ativa leitura do microfone. Use --no-mic-enabled para correr sem micro.",
    )
    parser.add_argument(
        "--no-mic-enabled",
        dest="mic_enabled",
        action="store_false",
        help="Desativa leitura do microfone (ruido fica 0).",
    )
    parser.add_argument(
        "--led-pin",
        type=int,
        default=27,
        help="GPIO do LED (BCM).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Intervalo de atualizacao em segundos.",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Ativa camera (necessaria para mostrar video/emocoes).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Mostra janela com frame, emocao, ruido e estado.",
    )
    parser.add_argument(
        "--resolution",
        default="640x480",
        help="Resolucao da camera (ex.: 640x480).",
    )
    parser.add_argument(
        "--wb-kelvin",
        type=int,
        default=None,
        help="Define temperatura de cor fixa em Kelvin (ex.: 4500). Se nao for usada, AWB fica automatico.",
    )
    parser.add_argument(
        "--color-gains",
        default=None,
        help="Gains de cor manual no formato R,B (ex.: 1.8,1.2). Desativa AWB se definido.",
    )
    parser.add_argument(
        "--ble-enabled",
        action="store_true",
        help="Ativa envio da cor da equipa para o LED via BLE.",
    )
    parser.add_argument(
        "--ble-device",
        default="BE:60:B4:00:48:B2",
        help="MAC address do controlador BLE (ex.: BE:60:B4:00:48:B2).",
    )
    parser.add_argument(
        "--ble-uuid",
        default=DEFAULT_BLE_UUID,
        help="UUID de escrita do controlador BLE.",
    )
    parser.add_argument(
        "--mode",
        choices=["debug", "normal", "demo"],
        default="normal",
        help="Modo de funcionamento da interface: debug, normal ou demo (menor ruido de logs).",
    )
    return parser.parse_args()


def parse_resolution(res_str: str) -> Tuple[int, int]:
    try:
        w, h = res_str.lower().split("x")
        return int(w), int(h)
    except Exception as exc:  # pragma: no cover - validacao simples
        raise argparse.ArgumentTypeError("Resolucao deve ser no formato LxA, ex.: 640x480") from exc


def parse_color_gains(gains_str: Optional[str]) -> Optional[Tuple[float, float]]:
    if gains_str is None:
        return None
    try:
        r_str, b_str = gains_str.split(",")
        return float(r_str), float(b_str)
    except Exception as exc:  # pragma: no cover - validacao simples
        raise argparse.ArgumentTypeError("Color gains deve ser 'R,B', ex.: 1.8,1.2") from exc


def main() -> None:
    args = parse_args()
    width, height = parse_resolution(args.resolution)
    color_gains = parse_color_gains(args.color_gains)
    backend = PiBackend(
        mic_device=args.mic_device,
        mic_samplerate=args.mic_samplerate,
        mic_frames=args.mic_frames,
        mic_gain=args.mic_gain,
        mic_noise_floor=args.mic_noise_floor,
        mic_smoothing=args.mic_smoothing,
        mic_auto_gain=args.mic_auto_gain,
        mic_target_level=args.mic_target_level,
        mic_peak_decay=args.mic_peak_decay,
        mic_max_gain=args.mic_max_gain,
        mic_enabled=args.mic_enabled,
        led_pin=args.led_pin,
        use_camera=args.camera or args.display,
        display=args.display,
        resolution=(width, height),
        wb_kelvin=args.wb_kelvin,
        color_gains=color_gains,
        ble_device=args.ble_device,
        ble_uuid=args.ble_uuid,
        ble_enabled=args.ble_enabled,
    )
    run_loop(backend, args.interval, args.display, args.mode)


if __name__ == "__main__":
    main()
