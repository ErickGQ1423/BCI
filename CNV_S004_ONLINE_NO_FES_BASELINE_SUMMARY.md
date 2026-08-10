# CNV S004 ONLINE — baseline sin FES

Fecha: 2026-07-09  
Sujeto: `CNV_PILOT_SUBJ_014`  
Sesión: `S004_ONLINE`  
Condición: MDM online + adaptive recentering acumulativo + endpoint validation + FES OFF

Configuración relevante:

```python
PREP_CONTROL_MODEL = "MDM"
RECENTERING = 1
M2_USE_SAVED_ADAPTIVE_RECENTER = True
SAVE_ADAPTIVE_T = True
ENDPOINT_VALIDATION_ENABLED = True
ENDPOINT_MDM_MI_THRESHOLD = 0.60
ENDPOINT_MDM_REST_THRESHOLD = 0.40
FES_toggle = 0
```

Nota importante:

- El Trial 1 falló sistemáticamente por `BAD_EEG` / `rms_too_high`.
- En los análisis finales conviene tratar el Trial 1 como warmup y excluirlo de métricas principales.

## Resumen de 5 corridas sin FES

| Corrida | Counter inicial | Updates final | Final Acc | Decision Acc | MI recall | REST recall | MDM original |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 25.0% | 29.4% | 50% | 0% | 25.0% |
| 2 | 0 | 5 | 45.0% | 50.0% | 80% | 10% | 45.0% |
| 3 | 5 | 6 | 45.0% | 64.3% | 60% | 30% | 55.0% |
| 4 | 6 | 15 | 70.0% | 82.4% | 70% | 70% | 75.0% |
| 5 | 15 | 21 | 60.0% | 63.2% | 80% | 40% | 60.0% |

## Promedios de las 5 corridas

```text
Final accuracy promedio      = 49.0%
Decision accuracy promedio   = 57.9%
MDM original promedio        = 52.0%
MI recall promedio           = 68.0%
REST recall promedio         = 30.0%
```

## Promedios quitando Trial 1

```text
Final accuracy promedio sin Trial 1 = 51.6%
MDM original promedio sin Trial 1   = 54.7%
```

## Últimas 2 corridas sin FES

Estas son las más relevantes porque ya tenían adaptación acumulativa más madura.

```text
Final accuracy promedio = 65.0%
Decision accuracy       = 72.8%
MI recall promedio      = 75.0%
REST recall promedio    = 55.0%
```

## Interpretación breve

- La adaptación acumulativa parece mejorar REST.
- REST pasó de 0–10% al inicio a 40–70% en las corridas finales.
- El MDM fue claramente más útil cuando el `adaptive_T.pkl` ya tenía varios updates.
- La siguiente comparación será contra 5 corridas con FES.

