import sys
import serial
from PyQt5.QtWidgets import QApplication, QMainWindow, QProgressBar, QVBoxLayout, QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

# ==========================================
# CONFIGURACIÓN ESTRICTA DEL HARDWARE
# ==========================================
PUERTO_SERIAL = 'COM17' 
BAUD_RATE = 115200

class TelemetriaLiDAR(QMainWindow):
    def _init_(self):
        super()._init_()

        # Configuración de la ventana principal
        self.setWindowTitle("HMI - Telemetría LiDAR Tiempo Real")
        self.setGeometry(100, 100, 350, 700)
        self.setStyleSheet("background-color: #0b0f19;") # Fondo oscuro profundo

        widget_central = QWidget()
        layout_principal = QVBoxLayout()
        widget_central.setLayout(layout_principal)
        self.setCentralWidget(widget_central)

        # 1. Título de la Interfaz
        self.label_titulo = QLabel("ESCÁNER DE RANGO 1D")
        self.label_titulo.setAlignment(Qt.AlignCenter)
        self.label_titulo.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.label_titulo.setStyleSheet("color: #38bdf8; margin-top: 20px; letter-spacing: 2px;")
        layout_principal.addWidget(self.label_titulo)

        # 2. Etiqueta numérica de la distancia
        self.label_distancia = QLabel("0 cm")
        self.label_distancia.setAlignment(Qt.AlignCenter)
        self.label_distancia.setFont(QFont("Consolas", 48, QFont.Bold))
        self.label_distancia.setStyleSheet("color: #ffffff; margin: 10px;")
        layout_principal.addWidget(self.label_distancia)

        layout_barra = QHBoxLayout()
        
        # 3. La Barra de Regla Vertical
        self.barra_distancia = QProgressBar()
        self.barra_distancia.setOrientation(Qt.Vertical)
        self.barra_distancia.setRange(0, 200)
        self.barra_distancia.setValue(0)
        self.barra_distancia.setTextVisible(False)
        
        # Estilo base de la barra (el color interior cambiará dinámicamente)
        self.estilo_base = """
            QProgressBar {
                border: 2px solid #1e293b;
                border-radius: 15px;
                background-color: #1e293b;
                width: 60px;
            }
        """
        self.barra_distancia.setStyleSheet(self.estilo_base + "QProgressBar::chunk { background-color: #3b82f6; border-radius: 13px; }")
        layout_barra.addWidget(self.barra_distancia)
        
        layout_principal.addLayout(layout_barra)
        layout_principal.setStretchFactor(layout_barra, 1)

        # ==========================================
        # INICIALIZACIÓN SERIAL Y BÚFER
        # ==========================================
        try:
            # Timeout en 0 es vital para leer sin bloquear el hilo principal
            self.ser = serial.Serial(PUERTO_SERIAL, BAUD_RATE, timeout=0)
            self.label_titulo.setText(f"ENLACE ACTIVO ({PUERTO_SERIAL})")
        except Exception as e:
            self.label_titulo.setText("ERROR DE PUERTO")
            self.label_titulo.setStyleSheet("color: #ef4444; margin-top: 20px; letter-spacing: 2px;")
            self.ser = None

        # Temporizador a 5 ms (200 Hz de tasa de refresco visual)
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_datos)
        self.timer.start(5) 

    def actualizar_datos(self):
        # Vaciado dinámico del búfer para latencia cero
        if self.ser and self.ser.in_waiting > 0:
            try:
                # Extraemos absolutamente todo lo acumulado en memoria
                datos_crudos = self.ser.read(self.ser.in_waiting)
                texto = datos_crudos.decode('utf-8', errors='ignore')
                lineas = texto.split('\n')
                
                # Leemos de atrás hacia adelante para quedarnos solo con el milisegundo actual
                for linea in reversed(lineas):
                    linea = linea.strip()
                    if linea.startswith("D,"):
                        valor_str = linea.split(',')[1]
                        distancia = int(valor_str)
                        
                        self.actualizar_interfaz(distancia)
                        break 
                        
            except Exception:
                pass

    def actualizar_interfaz(self, distancia):
        # Lógica de colores dinámicos según el nivel de proximidad
        if distancia < 50:
            color_hex = "#ef4444" # Rojo (Peligro/Muy cerca)
        elif distancia < 120:
            color_hex = "#eab308" # Amarillo (Precaución)
        else:
            color_hex = "#10b981" # Verde (Seguro/Lejos)

        # Aplicamos la distancia y el color renderizado
        self.label_distancia.setText(f"{distancia} cm")
        self.label_distancia.setStyleSheet(f"color: {color_hex}; margin: 10px;")
        
        estilo_dinamico = self.estilo_base + f"QProgressBar::chunk {{ background-color: {color_hex}; border-radius: 13px; }}"
        self.barra_distancia.setStyleSheet(estilo_dinamico)
        self.barra_distancia.setValue(distancia)

# Fíjate en los DOBLES guiones bajos: __name__ y __main__
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = TelemetriaLiDAR()
    ventana.show()
    sys.exit(app.exec_())