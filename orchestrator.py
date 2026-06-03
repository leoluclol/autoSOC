import os
import re
import time
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 1. CONFIGURAZIONE INIZIALE
# ==========================================
# Carica le variabili d'ambiente dal file .env (inclusa OPENAI_API_KEY)
load_dotenv()

# Inizializza il client OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sostituisci con i tuoi veri dati Kaggle presenti nel kernel-metadata.json
KAGGLE_USER_SLUG = "leonardoluchini/autoresearch-battery-soc" 

# ==========================================
# 2. FUNZIONI DI UTILITA'
# ==========================================
def read_file(filepath):
    """Legge il contenuto di un file testuale."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def write_file(filepath, content):
    """Scrive il contenuto in un file testuale."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def extract_python_code(ai_response):
    """Estrae solo il codice Python dalla risposta testuale dell'LLM"""
    # Usiamo una variabile per i backtick così l'IDE non si confonde 
    # e il copia-incolla non si rompe.
    ticks = "```"
    pattern = ticks + r"(?:python)?\n(.*?)" + ticks
    
    match = re.search(pattern, ai_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ai_response.strip()

def run_bash(command):
    """Esegue un comando nel terminale locale e restituisce l'output"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + "\n" + result.stderr

# ==========================================
# 3. INTERAZIONE CON KAGGLE CLOUD
# ==========================================
def run_kaggle_pipeline():
    """Esegue il push su Kaggle, attende la fine, e scarica i log"""
    print("🚀 Push del codice su Kaggle Cloud...")
    run_bash("kaggle kernels push -p . --accelerator NvidiaTeslaT4")
    
    print("⏳ Attesa del completamento del training (polling ogni 60s)...")
    while True:
        status_output = run_bash(f"kaggle kernels status {KAGGLE_USER_SLUG}")
        print(f"[Kaggle API] {status_output.strip()}")
        
        # Controlla se lo status indica che la run ha finito di girare
        if "complete" in status_output.lower() or "error" in status_output.lower() or "cancel" in status_output.lower():
            break
        time.sleep(60)
        
    print("📥 Download dei risultati...")
    run_bash("mkdir -p kaggle_output")
    # Pulisce i log vecchi per evitare di leggere output di run precedenti
    run_bash("rm -f ./kaggle_output/autoresearch-battery-soc.log") 
    
    # Questo comando scarica tutto il contenuto di /kaggle/working/ dentro ./kaggle_output/
    run_bash(f"kaggle kernels output {KAGGLE_USER_SLUG} -p ./kaggle_output")
    
    # Il file prodotto dal nostro script train.py si troverà qui:
    log_path = "./kaggle_output/autoresearch-battery-soc.log"
    
    if os.path.exists(log_path):
        return read_file(log_path)
    
    return "Errore: autoresearch-battery-soc.log non trovato nella cartella scaricata. Kaggle potrebbe essere andato in Timeout o in OOM prima di avviare Python."
# ==========================================
# 4. GESTIONE DELLE METRICHE
# ==========================================
def get_best_metric_from_tsv():
    """Legge il results.tsv per capire qual è il record (MAE) da battere"""
    if not os.path.exists("results.tsv"):
        write_file("results.tsv", "commit\ttest_mae\tmemory_gb\tstatus\tdescription\n")
        return float('inf')
    
    lines = read_file("results.tsv").strip().split('\n')[1:]
    best_mae = float('inf')
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2 and parts[1] != "0.0000":
            try:
                mae = float(parts[1])
                if mae < best_mae:
                    best_mae = mae
            except ValueError:
                continue
    return best_mae

# ==========================================
# 5. LOOP PRINCIPALE (L'AGENTE)
# ==========================================
def main_loop():
    program_instructions = read_file("program.md")
    
    # Inizializza la memoria dell'agente
    conversation_history = [
        {"role": "system", "content": program_instructions},
        {"role": "user", "content": "Iniziamo l'esperimento. Leggi il train.py attuale e proponi la tua prima modifica strutturale. Rispondi SOLO con il nuovo codice completo dentro un blocco ```python"}
    ]
    
    iteration = 1
    while True:
        print(f"\n{'='*40}\n🔄 ITERAZIONE {iteration}\n{'='*40}")
        current_train_py = read_file("train.py")
        
        # 1. Chiediamo all'LLM di pensare e scrivere il nuovo codice
        print("🧠 L'Agente sta pensando e modificando l'architettura...")
        prompt = f"Ecco il codice attuale di train.py:\n```python\n{current_train_py}\n```\nQual è la tua prossima mossa?"
        conversation_history.append({"role": "user", "content": prompt})
        
        # Chiamata API all'LLM
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=conversation_history,
            temperature=0.7
        )
        ai_message = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_message})
        
        # 2. Estraiamo e salviamo il codice, poi facciamo il commit
        new_code = extract_python_code(ai_message)
        write_file("train.py", new_code)
        
        commit_msg = f"Auto-run iterazione {iteration}"
        run_bash(f'git commit -am "{commit_msg}"')
        commit_hash = run_bash("git rev-parse --short HEAD").strip()
        
        # 3. Lanciamo la pipeline su Kaggle
        run_log = run_kaggle_pipeline()

        print(run_log)
        
        # 4. Valutiamo i risultati estraendo le metriche dal log
        test_mae_match = re.search(r"test_mae_percent:\s+([0-9.]+)", run_log)
        vram_match = re.search(r"peak_vram_mb:\s+([0-9.]+)", run_log)
        
        current_mae = float(test_mae_match.group(1)) if test_mae_match else 0.0
        vram_gb = float(vram_match.group(1))/1024 if vram_match else 0.0
        
        best_mae = get_best_metric_from_tsv()
        
        if current_mae == 0.0:
            status = "crash"
            print("❌ Il training su Kaggle è andato in crash.")
            run_bash("git reset HEAD~1 --hard") # Rollback immediato
        elif current_mae < best_mae:
            status = "keep"
            print(f"✅ MIGLIORAMENTO! Nuovo MAE: {current_mae} (precedente: {best_mae})")
            # Nessun rollback, manteniamo il commit
        else:
            status = "discard"
            print(f"📉 Nessun miglioramento (MAE: {current_mae}). Rollback in corso...")
            run_bash("git reset HEAD~1 --hard") # Rollback
            
        # 5. Salviamo nel TSV
        tsv_line = f"{commit_hash}\t{current_mae:.4f}\t{vram_gb:.1f}\t{status}\tIterazione {iteration}\n"
        with open("results.tsv", "a") as f:
            f.write(tsv_line)
            
        # 6. Diamo il feedback all'LLM per il prossimo loop
        # Passiamo solo gli ultimi 1500 caratteri del log per risparmiare token
        feedback = f"Risultato della run su Kaggle:\n{run_log[-1500:]}\nStatus: {status}."
        if status == "crash":
            feedback += "\nIl codice ha generato un errore o Kaggle è andato in OOM. Correggi il bug o semplifica l'architettura."
        elif status == "discard":
            feedback += "\nLe tue modifiche hanno peggiorato o non migliorato le metriche. Il codice è stato ripristinato alla versione precedente. Prova un'altra strada."
            
        conversation_history.append({"role": "user", "content": feedback})
        iteration += 1

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n⏹️ Esecuzione interrotta manualmente dall'utente.")