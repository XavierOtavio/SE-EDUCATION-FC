import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyaudio
from gpiozero import Button, LED

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover
    Picamera2 = None

NOISE_MIN = 0
NOISE_MAX = 1023
THRESHOLD = 512
WINDOW_NAME = "Estadio Interativo"


@dataclass
class FaceEmotion:
    x: int
    y: int
    w: int
    h: int
    emotion: str


class EmotionDetector:
    """Deteta faces e estima emocao simples (sorriso -> Feliz; senao Neutro)."""

    def __init__(self) -> None:
        cascades = self._cascade_dir()
        self.face_cascade = cv2.CascadeClassifier(
            str(cascades / "haarcascade_frontalface_default.xml")
        )
        self.smile_cascade = cv2.CascadeClassifier(
            str(cascades / "haarcascade_smile.xml")
        )
        if self.face_cascade.empty() or self.smile_cascade.empty():
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
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        results: List[FaceEmotion] = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y : y + h, x : x + w]
            smiles = self.smile_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.7, minNeighbors=20
            )
            emotion = "Feliz" if len(smiles) > 0 else "Neutro"
            results.append(FaceEmotion(x, y, w, h, emotion))
        return results


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

    def read_level(self) -> int:
        data = self.stream.read(self.frames, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16)
        if samples.size == 0:
            return 0
        rms = float(np.sqrt(np.mean(np.square(samples))))
        rms_norm = rms / self.max_int
        if rms_norm > self.noise_floor:
            rms_norm = (rms_norm - self.noise_floor) / max(1e-6, (1 - self.noise_floor))
        else:
            rms_norm = 0.0
        if self.auto_gain:
            self.rolling_peak = max(rms_norm, self.rolling_peak * self.peak_decay)
            dyn_gain = self.gain
            if self.rolling_peak > 1e-6:
                dyn_gain *= self.target_level / self.rolling_peak
            dyn_gain = min(dyn_gain, self.max_gain)
        else:
            dyn_gain = self.gain
        level_raw = int(min(NOISE_MAX, rms_norm * dyn_gain * NOISE_MAX))
        if self.smoothing <= 0:
            level_smooth = level_raw
        else:
            if self.prev_level is None:
                level_smooth = level_raw
            else:
                alpha = self.smoothing
                level_smooth = int(self.prev_level + alpha * (level_raw - self.prev_level))
        self.prev_level = level_smooth
        return level_smooth

    def close(self) -> None:
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.pa.terminate()


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
        button_pin: int,
        led_pin: int,
        use_camera: bool,
        display: bool,
        resolution: Tuple[int, int],
        wb_kelvin: Optional[int],
        color_gains: Optional[Tuple[float, float]],
    ) -> None:
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
        self.button = Button(button_pin, pull_up=False)
        self.led = LED(led_pin)

        self.detector: Optional[EmotionDetector] = None
        self.camera: Optional[Picamera2] = None
        self.display = display

        if use_camera or display:
            if Picamera2 is None:
                raise RuntimeError(
                    "picamera2 nao instalado; instale python3-picamera2 ou desative --camera/--display."
                )
            self.camera = Picamera2()
            self.camera.set_controls({"AwbEnable": True, "AwbMode": 1}) 
            config = self.camera.create_preview_configuration(
                main={"size": resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            if wb_kelvin or color_gains:
                controls = {"AwbEnable": False}
                if wb_kelvin:
                    controls["ColourTemperature"] = wb_kelvin
                if color_gains:
                    controls["ColourGains"] = color_gains
                self.camera.set_controls(controls)
            else:
                # Mantem AWB ligado se nao houver configuracao manual.
                self.camera.set_controls({"AwbEnable": True})
            self.detector = EmotionDetector()

        time.sleep(0.05)

    def read(self) -> Tuple[int, bool]:
        """Ler ruido (0-1023) do microfone USB e estado do botao (True/False)."""
        noise_scaled = self.audio.read_level()
        pressure_val = self.button.is_pressed
        return noise_scaled, bool(pressure_val)

    def set_led(self, on: bool) -> None:
        self.led.on() if on else self.led.off()

    def capture_frame(self):
        if not self.camera:
            return None
        return self.camera.capture_array("main")

    def detect_emotions(self, frame_bgr) -> List[FaceEmotion]:
        if not self.detector or frame_bgr is None:
            return []
        return self.detector.detect(frame_bgr)

    def close(self) -> None:
        if self.camera:
            self.camera.stop()
        self.button.close()
        self.led.close()
        self.audio.close()


def decide(noise: int, pressure: bool) -> Tuple[str, bool]:
    """Retorna (mensagem, led_on) segundo os criterios."""
    if noise > THRESHOLD and pressure:
        return "GOLO", True
    if noise > THRESHOLD and not pressure:
        return "VAIA", False
    return "Entusiasmo normal", False


def draw_overlay(
    frame, faces: List[FaceEmotion], noise: int, pressure: bool, message: str
) -> None:
    """Desenha bounding boxes, emocao e info de ruido/pressao/estado no frame."""
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
    overlay_text = f"Ruido: {noise}  Pressao: {pressure}  Estado: {message}"
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


def run_loop(backend: PiBackend, interval: float, display: bool) -> None:
    print("A ler sensores no Raspberry Pi. Janela de video se display ativo. Ctrl+C para sair.")
    try:
        while True:
            noise, pressure = backend.read()
            message, led_on = decide(noise, pressure)
            backend.set_led(led_on)

            frame = backend.capture_frame() if display else None
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Camera montada invertida: rodar 180 graus para corrigir orientacao.
                frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
                faces = backend.detect_emotions(frame_bgr)
                draw_overlay(frame_bgr, faces, noise, pressure, message)
                cv2.imshow(WINDOW_NAME, frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                led_label = "ON " if led_on else "OFF"
                print(
                    f"Ruido={noise:4d} | Pressao={pressure!s:5} | LED={led_label} | {message}"
                )
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nTerminando...")
    finally:
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
        "--button-pin",
        type=int,
        default=17,
        help="GPIO do botao (BCM). Use pull-down externo ou ajuste wiring.",
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
        help="Mostra janela com frame, emocao, ruido, pressao e estado.",
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
        button_pin=args.button_pin,
        led_pin=args.led_pin,
        use_camera=args.camera or args.display,
        display=args.display,
        resolution=(width, height),
        wb_kelvin=args.wb_kelvin,
        color_gains=color_gains,
    )
    run_loop(backend, args.interval, args.display)


if __name__ == "__main__":
    main()

