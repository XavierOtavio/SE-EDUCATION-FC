import argparse
import time
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
        # Heuristica simples: se esta escuro ou olhos nao aparecem, marcar como Zangado.
        if not has_eyes or mean_intensity < 90:
            return "Zangado"
        return "Triste"


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
                return self.pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=frames,
                    input_device_index=device_index,
                )
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

    def read_level(self) -> int:
        return 0


class BLEController:
    """Envia cor via BLE com conexão persistente. Apenas envia quando necessário."""

    def __init__(self, device_mac: str, write_uuid: str, enabled: bool) -> None:
        self.device_mac = device_mac
        self.write_uuid = write_uuid
        self.enabled = enabled and BleakClient is not None
        self.last_color: Optional[Tuple[int, int, int]] = None
        self.client: Optional["BleakClient"] = None
        self.connection_lock = threading.Lock()
        self.connect_thread: Optional[threading.Thread] = None
        self.running = True
        
        if enabled and BleakClient is None:
            print("[BLE] bleak nao instalado; desative BLE ou instale bleak.")
        elif self.enabled:
            # Iniciar thread de conexão persistente
            self.connect_thread = threading.Thread(target=self._maintain_connection, daemon=True)
            self.connect_thread.start()

    async def _ensure_client(self) -> bool:
        """Garante que o cliente BLE está conectado"""
        with self.connection_lock:
            if self.client and self.client.is_connected:
                return True
            try:
                print(f"[BLE] A ligar a {self.device_mac}...")
                self.client = BleakClient(self.device_mac)
                await self.client.connect()
                if not self.client.is_connected:
                    print("[BLE] Falha ao ligar. Verifique o controlador.")
                    self.client = None
                    return False
                print("[BLE] Ligado com sucesso.")
                return True
            except Exception as exc:
                print(f"[BLE] Erro ao ligar: {exc}")
                self.client = None
                return False

    def _maintain_connection(self) -> None:
        """Thread para manter conexão BLE ativa continuamente"""
        while self.running:
            try:
                asyncio.run(self._maintain_connection_async())
            except Exception as e:
                print(f"[BLE] Erro na thread de manutenção: {e}")
            time.sleep(5)  # Tentar reconectar a cada 5 segundos se desligado

    async def _maintain_connection_async(self) -> None:
        """Manter conexão ativa"""
        if not self.enabled:
            return
        await self._ensure_client()

    async def _send_async(self, r: int, g: int, b: int) -> None:
        """Enviar cor via BLE (conexão já deve estar ativa)"""
        if not self.enabled:
            return
        if not await self._ensure_client():
            return
        try:
            cmd = bytes([0x7E, 0x00, 0x05, 0x03, r, g, b, 0x00, 0xEF])
            await self.client.write_gatt_char(self.write_uuid, cmd)
            print(f"[BLE] Cor enviada (R={r} G={g} B={b}).")
        except Exception as exc:
            print(f"[BLE] Erro ao enviar cor: {exc}")
            with self.connection_lock:
                self.client = None

    def send_color(self, bgr: Tuple[int, int, int]) -> None:
        """Enviar cor apenas se for diferente da anterior"""
        if not self.enabled:
            return
        if bgr == self.last_color:
            return  # Não enviar se for a mesma cor
        self.last_color = bgr
        b, g, r = bgr
        try:
            asyncio.run(self._send_async(r, g, b))
        except RuntimeError as exc:
            print(f"[BLE] Erro ao correr loop asyncio: {exc}")
            with self.connection_lock:
                self.client = None

    # Override com retry para evitar falha na primeira ligação/envio
    def send_color(self, bgr: Tuple[int, int, int]) -> None:
        """Enviar cor apenas se for diferente da anterior, com 2 tentativas."""
        if not self.enabled:
            return
        if bgr == self.last_color:
            return
        self.last_color = bgr
        b, g, r = bgr
        for attempt in range(2):
            try:
                asyncio.run(self._send_async(r, g, b))
                return
            except RuntimeError as exc:
                print(f"[BLE] Erro ao correr loop asyncio (tentativa {attempt+1}): {exc}")
                with self.connection_lock:
                    self.client = None
                time.sleep(0.2)
        print("[BLE] Nao foi possivel enviar cor apos retries.")

    def turn_off_leds(self) -> None:
        """Desligar LEDs (enviar preto)"""
        self.send_color((0, 0, 0))

    async def _disconnect_async(self) -> None:
        """Desligar BLE"""
        with self.connection_lock:
            if self.client and self.client.is_connected:
                try:
                    await self.client.disconnect()
                    print("[BLE] Desligado.")
                except Exception as exc:
                    print(f"[BLE] Erro ao desligar: {exc}")
            self.client = None

    def close(self) -> None:
        """Fechar conexão BLE e thread de manutenção"""
        if not self.enabled:
            return
        self.running = False
        if self.connect_thread:
            self.connect_thread.join(timeout=2)
        try:
            asyncio.run(self._disconnect_async())
        except RuntimeError:
            pass


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
        
        # Buffer para frames e emoções
        self.frame_buffer = deque(maxlen=3)
        self.frame_lock = threading.Lock()
        self.capture_thread: Optional[threading.Thread] = None
        self.camera_running = False
        
        # Armazenar cor da equipa para usar quando LEDs forem ativados
        self.team_color: Tuple[int, int, int] = (0, 0, 255)  # Vermelho por defeito
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
        print(f"[LEDs] Ativados com cor da equipa (RGB={color})")

    def deactivate_leds(self) -> None:
        """Desativar LEDs"""
        self.ble.turn_off_leds()
        print("[LEDs] Desativados")


def decide(noise: int) -> Tuple[str, bool]:
    """Retorna (mensagem, led_on) segundo os criterios apenas de ruido."""
    if noise > THRESHOLD:
        return "Ruido alto", True
    return "Ruido normal", False


def classify_sound(noise_level: int, peak_freq: float, amp_duration: float) -> str:
    """
    Usa amplitude, pico de frequencia e tempo de som mantido para distinguir
    gritos (golo) de vaias graves e de conversa normal.
    """
    print(
        f"Peak Frequency: {peak_freq:.1f} Hz, Noise Level: {noise_level}, Duracao: {amp_duration:.2f}s"
    )
    # Limiares calibrados com os valores que indicou:
    # - Golo: freq > 500 Hz e ruido > 600 (ou pico alto prolongado)
    # - Vaia: freq 150-250 Hz e ruido ~300
    # - Conversa: freq 50-500 Hz e ruido 100-300 (neutro)
    if (peak_freq > 500 and noise_level > 550) or (
        amp_duration > 0.8 and peak_freq > 450 and noise_level > 500
    ):
        return "Som: Grito prolongado", "golo"
    if (150 <= peak_freq <= 320 and noise_level >= 240 and amp_duration > 0.6) or (
        noise_level > 320 and 120 < peak_freq < 400
    ):
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


def run_loop(backend: PiBackend, interval: float, display: bool) -> None:
    print("A ler sensores no Raspberry Pi. Janela de video se display ativo. Ctrl+C para sair.")
    print("Pressione 'L' para ativar LEDs ou 'K' para desativar. Pressione 'Q' para sair.")
    
    last_ble_color: Optional[Tuple[int, int, int]] = None
    override_color: Optional[Tuple[int, int, int]] = None
    override_until: Optional[float] = None
    amp_duration = 0.0
    last_ts = time.time()
    amp_active = False
    high_count = 0
    low_count = 0
    display_buffer = ImageDisplayBuffer(buffer_size=3, display_interval=0.5)
    leds_active = False
    
    try:
        while True:
            now = time.time()
            # Limpar override se expirou
            if override_until and now > override_until:
                override_color = None
                override_until = None

            dt = now - last_ts
            last_ts = now

            noise, peak_freq = backend.read()
            # Atualizar duração de som ativo: sequência de amostras "altas" conta como um bloco.
            # Use limiares alinhados com os valores medidos no estádio.
            high_thresh = 200  # ruido acima disto sugere evento relevante
            low_thresh = 120   # abaixo disto consideramos que terminou
            is_high = noise > high_thresh or peak_freq > 500
            if is_high:
                high_count += 1
                low_count = 0
            else:
                low_count += 1
                high_count = 0
            # Histerese simples: precisa de 2 leituras altas para ligar, 2 baixas para desligar.
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
            
            # Detetar som e preparar override (mas não ativar LEDs automaticamente)
            if sound_kind == "golo":
                override_color = (0, 255, 0)  # verde em BGR
                override_until = now + 5.0
            elif sound_kind == "vaia":
                override_color = (0, 0, 255)  # vermelho em BGR
                override_until = now + 5.0

            # Obter frame do buffer (não bloqueia)
            frame_data = backend.get_latest_frame() if display else None
            
            if frame_data is not None:
                frame_rgb = frame_data['frame']
                faces = backend.detect_emotions(frame_rgb)
                team_color = dominant_color(frame_rgb, faces)
                backend.update_team_color(team_color)
                
                # Desenhar overlay
                draw_overlay(frame_rgb, faces, noise, message, team_color, sound_label)
                display_buffer.add_frame(frame_rgb, {'faces': faces, 'sound': sound_label})
                
                # Exibir apenas a cada intervalo
                if display_buffer.display_if_needed(WINDOW_NAME):
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == ord("Q"):
                        break
                    elif key == ord("l") or key == ord("L"):
                        # Ativar LEDs
                        backend.activate_leds()
                        leds_active = True
                    elif key == ord("k") or key == ord("K"):
                        # Desativar LEDs
                        backend.deactivate_leds()
                        leds_active = False
            else:
                led_label = "ON " if led_on else "OFF"
                print(f"Ruido={noise:4d} | LED={led_label} | {message} | {sound_label}")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nTerminando...")
    finally:
        backend.deactivate_leds()  # Desligar LEDs ao sair
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
    run_loop(backend, args.interval, args.display)


if __name__ == "__main__":
    main()


