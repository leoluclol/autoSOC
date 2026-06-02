# Modifiche ai parametri iperperimetrali per ridurre l'uso della memoria
DEPTH = 2                # riduce ulteriormente il numero di strati transformer
DEVICE_BATCH_SIZE = 32   # riduce ulteriormente la dimensione del batch per dispositivo
MAX_SEQ_LEN = 512        # riduce la lunghezza massima della sequenza per ridurre la memoria

# Il resto del codice rimane invariato