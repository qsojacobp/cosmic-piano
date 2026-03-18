## Requirements  
### System dependency (install BEFORE installing anything through pip)  
**macOS:**  
```brew install portmidi```

**Linux (Debian/Ubuntu):**  
```sudo apt install libportmidi-dev portmidi```

**Windows:**  
Download and install portmidi directly via https://portmedia.sourceforge.net/

**Python packages:**  
```python3 -m pip install -r requirements.txt```

Python 3.9 or higher is recommended.

## Usage  
```python cosmic-piano.py --ARGS```  

### Valid ARGS  
```--generate``` - build audio samples, required before anything can be played  
```--generate --duration n``` - builds sample of length n seconds  
```--play``` - play via MIDI device, will automatically detect connected device  
```--keyboard``` - play via computer keyboard  
```--list``` - lists objects and data sources  
```--preview 67``` - previews single note mapped to specified number  

## Currently supported objects  
GW150914 (black hole merger)
Gw170817 (neutron star merger)
GW190521 (massive black hole merger)
Vela Pulsar
LGM-1 (first discovered pulsar)
PSR B1937+21 (millisecond pulsar)
Crab Pulsar
OBAFGKM stars (incl. Sirius, the Sun, Proxima Centauri)
NGC 3198 (Milky Way-like galaxy)
M87
Arp 220
3C 273 (bright quasar)
Radio emissions from Jupiter
Magnetosphere "whistlers" from Saturn
"Chorus waves" from Earth