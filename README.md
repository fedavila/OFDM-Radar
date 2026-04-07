# OFDM Radar Simulation

Python-based simulation of an **OFDM radar system** for range–Doppler estimation and target detection, inspired by modern **Joint Sensing and Communication (JSC)** concepts.

---

## Overview

This project implements a complete OFDM radar processing chain:

- OFDM signal generation (modulation, IFFT, CP)  
- Target modeling (range, velocity, RCS)  
- Received signal processing  
- Range–Doppler map (2D periodogram)  
- Target detection (Pfa-based threshold)  

The estimation is based on identifying **sinusoids in time–frequency space**, a key principle in OFDM radar.

---

## Example Output

![Range-Doppler Map](results/2D_periodogram.png)

---

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python simulation.py


