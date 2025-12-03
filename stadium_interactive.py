import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from gpiozero import MCP3008, Button, LED

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - opcional em ambientes sem camera
    Picamera2 = None

NOISE_MIN = 0
NOISE_MAX = 1023
THRESHOLD = 512


class PiBackend:
    """
    Le sensores reais no Raspberry Pi (MCP3008 para analógico) e controla LED e camera.

    Mapeamento esperado:
    - MCP3008 canal 0: sensor de ruido (analogico)
    - GPIO 17: botao (digital, pull-down externo ou pull_up=True se usar pull-up)
    - GPIO 27: LED (digital)
    - Camera Pi v2: opcional (capturas quando GOLO/VAIA mudam)
    """

    def __init__(
        self,
        noise_channel: int,
        button_pin: int,
        led_pin: int,
        camera_enabled: bool,
        camera_dir: str,
        attempts: int = 50,
    ) -> None:
        self.noise_sensor = MCP3008(channel=noise_channel)
        self.button = Button(button_pin, pull_up=False)
        self.led = LED(led_pin)

        self.camera_dir = Path(camera_dir)
        self.camera: Optional[Picamera2] = None
        if camera_enabled:
            if Picamera2 is None:
                raise RuntimeError(
                    "picamera2 nao instalado; instale python3-picamera2 ou desative --camera."
                )
            self.camera_dir.mkdir(parents=True, exist_ok=True)
            self.camera = Picamera2()
            self.camera.configure(self.camera.create_still_configuration())
            self.camera.start()

        self.max_attempts = attempts
        time.sleep(0.1)

    def read(self) -> Tuple[int, bool]:
        """
        Ler ruido (0-1023) e estado do botao (True/False).
        MCP3008.value devolve 0.0-1.0; escalamos para 0-1023.
        """
        for _ in range(self.max_attempts):
            noise_val = self.noise_sensor.value
            pressure_val = self.button.is_pressed
            if noise_val is not None:
                noise_scaled = int(round(noise_val * NOISE_MAX))
                return noise_scaled, bool(pressure_val)
            time.sleep(0.01)
        raise RuntimeError("Sem leitura do MCP3008; confirme SPI e ligações.")

    def set_led(self, on: bool) -> None:
        self.led.on() if on else self.led.off()

    def capture(self, message: str) -> None:
        """Captura uma foto quando a camera esta ativa."""
        if not self.camera:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.camera_dir / f"{timestamp}_{message.lower()}.jpg"
        self.camera.capture_file(str(filename))
        print(f"[camera] captura guardada em {filename}")

    def close(self) -> None:
        if self.camera:
            self.camera.stop()
        self.noise_sensor.close()
        self.button.close()
        self.led.close()


def decide(noise: int, pressure: bool) -> Tuple[str, bool]:
    """Retorna (mensagem, led_on) segundo os criterios."""
    if noise > THRESHOLD and pressure:
        return "GOLO", True
    if noise > THRESHOLD and not pressure:
        return "VAIA", False
    return "Entusiasmo normal", False


def run_loop(backend: PiBackend, interval: float) -> None:
    print("A ler sensores no Raspberry Pi. Ctrl+C para sair.")
    last_message: Optional[str] = None
    try:
        while True:
            noise, pressure = backend.read()
            message, led_on = decide(noise, pressure)
            backend.set_led(led_on)
            if message != last_message and message in {"GOLO", "VAIA"}:
                backend.capture(message)
            last_message = message
            led_label = "ON " if led_on else "OFF"
            print(
                f"Ruido={noise:4d} | Pressao={pressure!s:5} | LED={led_label} | {message}"
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nTerminando...")
    finally:
        backend.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estadio interativo em Raspberry Pi (GPIO + MCP3008 + camera opcional)."
    )
    parser.add_argument(
        "--noise-channel",
        type=int,
        default=0,
        help="Canal do MCP3008 para o sensor de ruido (0-7).",
    )
    parser.add_argument(
        "--button-pin",
        type=int,
        default=17,
        help="GPIO do botao (BCM). Use pull-down externo ou troque pull_up conforme wiring.",
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
        help="Ativa capturas com a Camera Pi v2 quando a mensagem muda para GOLO/VAIA.",
    )
    parser.add_argument(
        "--camera-dir",
        default="capturas",
        help="Diretorio onde guardar fotos (quando camera ativa).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = PiBackend(
        noise_channel=args.noise_channel,
        button_pin=args.button_pin,
        led_pin=args.led_pin,
        camera_enabled=args.camera,
        camera_dir=args.camera_dir,
    )
    run_loop(backend, args.interval)


if __name__ == "__main__":
    main()
