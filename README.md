# Multi-Modal Bioprocess Digital Twin

A local Streamlit application that demonstrates a simplified, illustrative model of:
- biomass growth,
- substrate consumption,
- product formation,
- four hypothetical physical-field input channels,
- a growth-to-production switch,
- a simplified CRISPRi/resource-allocation representation,
- baseline versus multi-field comparison.

## 1. Install Python
Install Python 3.10 or newer from https://www.python.org/downloads/.
During Windows installation, enable **Add Python to PATH**.

## 2. Open the project folder
Extract the ZIP file. Open Command Prompt or PowerShell in the extracted folder.

## 3. Create an environment (recommended)
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

## 4. Install dependencies
```bash
python -m pip install -r requirements.txt
```

## 5. Run the application
```bash
streamlit run app.py
```

Your browser should open the local application. If it does not, copy the local address shown in the terminal (usually http://localhost:8501) into your browser.

## How to explore
1. Keep the initial settings and open **Live simulation**.
2. Change one physical-field slider at a time.
3. Compare the final product concentration with the no-field baseline.
4. Open **Genetic control** to inspect each normalized actuator response.
5. Change broth turbidity to see the illustrative reduction in optical response.
6. Download the generated time-series data from **Export results**.

## Scientific note
This is a computational concept demonstrator, not a validated biological digital twin. The four field-response functions and coupling factors are illustrative. The application does not prove that the four actuator systems are experimentally available, orthogonal, safe, or effective in a real production strain. The current model is well-mixed and omits many important effects, including spatial gradients, oxygen-transfer kinetics, cell death, maintenance metabolism, actuator heating, electrochemical side reactions, cavitation damage, and validated magnetic actuation mechanisms.
