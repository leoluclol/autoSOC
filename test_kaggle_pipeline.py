import os
import time
import subprocess
import shutil

# Il tuo slug univoco di Kaggle
KAGGLE_USER_SLUG = "leonardoluchini/autoresearch-battery-soc"

def run_bash(command):
    """Esegue un comando nel terminale locale e restituisce l'output"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + "\n" + result.stderr

def main():
    print("=== 🔍 INIZIO DIAGNOSTICA PIPELINE KAGGLE ===")
    
    # 1. Verifica che i metadati esistano
    if not os.path.exists("kernel-metadata.json"):
        print("❌ Errore: 'kernel-metadata.json' non trovato nella cartella corrente!")
        return
    print("✅ 'kernel-metadata.json' rilevato con successo.")

    # 2. Backup del train.py reale per non perdere il tuo lavoro
    has_backup = False
    if os.path.exists("train.py"):
        print("📦 Salvataggio di backup di 'train.py' in 'train.py.bak'...")
        shutil.copy("train.py", "train.py.bak")
        has_backup = True

    # 3. Scrittura di uno script finto di test (Dummy Script)
    test_script_content = """import sys
import os
import time

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Attiviamo il logging su file richiesto
sys.stdout = DualLogger("run.log")
sys.stderr = sys.stdout

print("=== COMINCIA LA RUN DI PROVA DI SINCERAMENTO ===")
print("Esecuzione ambiente Kaggle Cloud riuscita con successo.")
print("Attesa di simulazione computazionale...")
time.sleep(3)

# Generiamo esattamente la struttura di metriche attesa dall'orchestratore
print("\\n========================================")
print("FINAL EVALUATION METRICS")
print("========================================")
print("test_mae_percent:   1.2345")
print("test_rmse_percent:  2.5555")
print("max_error_percent:  3.9999")
print("----------------------------------------")
print("training_seconds:   3.0")
print("total_seconds:      6.0")
print("peak_vram_mb:       512.0")
print("num_steps:          5")
print("========================================")
print("Fine del processo simulato di training.")
"""

    print("📝 Generazione del file 'train.py' temporaneo di test...")
    with open("train.py", "w", encoding="utf-8") as f:
        f.write(test_script_content)

    try:
        # 4. Push del codice su Kaggle
        print("🚀 Esecuzione: kaggle kernels push...")
        push_out = run_bash("kaggle kernels push -p .")
        print(f"[Kaggle CLI Push]:\n{push_out.strip()}\n")

        # 5. Monitoraggio dello stato (Polling)
        print("⏳ Polling dello stato del Cloud Kaggle (ogni 15 secondi)...")
        while True:
            status_out = run_bash(f"kaggle kernels status {KAGGLE_USER_SLUG}")
            status_clean = status_out.strip()
            print(f"[Kaggle Status]: {status_clean}")
            
            if "complete" in status_clean.lower() or "error" in status_clean.lower() or "cancel" in status_clean.lower():
                break
            time.sleep(15)

        # 6. Download dei Risultati della working directory
        print("\n📥 Scaricamento dell'output (/kaggle/working/)...")
        os.makedirs("kaggle_test_output", exist_ok=True)
        
        # Pulizia di vecchi residui di test
        if os.path.exists("kaggle_test_output/run.log"):
            os.remove("kaggle_test_output/run.log")
            
        output_out = run_bash(f"kaggle kernels output {KAGGLE_USER_SLUG} -p ./kaggle_test_output")
        print(f"[Kaggle CLI Output]:\n{output_out.strip()}\n")

        # 7. Ispezione del file scaricato
        log_path = "./kaggle_test_output/run.log"
        print("=========================================")
        print("🔍 VERIFICA DEL RISULTATO GENERATO")
        print("=========================================")
        if os.path.exists(log_path):
            print("🎉 EVVIVA! Il file 'run.log' è stato estratto correttamente.")
            print("Ecco il contenuto recuperato direttamente dal Cloud:")
            print("-" * 50)
            with open(log_path, "r", encoding="utf-8") as log_file:
                print(log_file.read())
            print("-" * 50)
        else:
            print("❌ ERRORE CRITICO: Il file 'run.log' non è presente nella cartella scaricata.")
            print("I file effettivamente scaricati da Kaggle sono:")
            print(os.listdir("kaggle_test_output") if os.path.exists("kaggle_test_output") else "Nessuno (Cartella vuota)")

    except Exception as e:
        print(f"Si è verificato un errore imprevisto durante il test: {e}")

    finally:
        # 8. Ripristino del tuo file train.py di ricerca originale
        print("\n🧹 Fasi di ripristino dell'ambiente...")
        if has_backup:
            print("📦 Ripristino del file 'train.py' originale dal backup.")
            shutil.copy("train.py.bak", "train.py")
            os.remove("train.py.bak")
        elif os.path.exists("train.py"):
            os.remove("train.py")
        print("=== 🏁 DIAGNOSTICA TERMINATA ===")

if __name__ == "__main__":
    main()