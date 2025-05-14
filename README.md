# audio-synthesis-service
STT server that synthesizes audio from text. 
Note the original synthesis strategy implemented and testing can be found in ./original_synthesis_server.py.
The new and improved strategy with dynamic synthesis speed can be found in ./synthesis_server.py

To run the service you just install all requirements in requirements.txt into a virtual environment.
Then run:

 source ./ubuntu_venv/bin/activate
 python3 synthesis_server.py

# Currently:
1. Using windows and venv_synthesis (require torch < 2.6)
2.  python .\synthesis_server.py (not python3)