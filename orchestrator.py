import os
import re
import time
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 1. INITIAL CONFIG
# ==========================================
load_dotenv()

# Loads openai client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project="proj_F4j8NgAkr8blWRXc4hqHDgAZ"
)

# Kaggle dataset location
KAGGLE_USER_SLUG = "leonardoluchini/autoresearch-battery-soc" 

# ==========================================
# 2. UTILITIES
# ==========================================
def read_file(filepath):
    """Reads a text file's content"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def write_file(filepath, content):
    """Writes content to a file"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def extract_python_code(ai_response):
    """Extracts python code from the LLM response"""
    ticks = "```" # Avoids breaking when talking to LLMs
    pattern = ticks + r"(?:python)?\n(.*?)" + ticks
    
    match = re.search(pattern, ai_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ai_response.strip()

def extract_llm_text(ai_response):
    """Extracts everything from the LLM response EXCEPT the python code blocks"""
    ticks = "```"
    # Matches the code blocks including the backticks
    pattern = ticks + r"(?:python)?\n.*?" + ticks
    
    # Replace code blocks with an empty string (or a newline)
    cleaned_text = re.sub(pattern, "", ai_response, flags=re.DOTALL)
    
    # Clean up any leftover double-newlines or trailing spaces
    return re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()

def run_bash(command):
    """Executes a bash command and returns the output"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + "\n" + result.stderr

# ==========================================
# 3. KAGGLE CLOUD INTERACTION
# ==========================================
def run_kaggle_pipeline():
    """Pushes code to Kaggle for training and waits for the end of training log"""
    run_bash("kaggle kernels push -p . --accelerator NvidiaTeslaT4")
    
    print("Training in progress")
    while True:
        status_output = run_bash(f"kaggle kernels status {KAGGLE_USER_SLUG}")
        print(f"[Kaggle API] {status_output.strip()}")
        
        # Checks run status
        if "complete" in status_output.lower() or "error" in status_output.lower() or "cancel" in status_output.lower():
            break
        time.sleep(30)
        
    print("Log download")
    run_bash("mkdir -p kaggle_output") # Creates log directory if it does not exist
    run_bash("rm -f ./kaggle_output/autoresearch-battery-soc.log") # Cleans log directory
    run_bash(f"kaggle kernels output {KAGGLE_USER_SLUG} -p ./kaggle_output") # Downloads new log
    log_path = "./kaggle_output/autoresearch-battery-soc.log"
    
    if os.path.exists(log_path):
        return read_file(log_path)
    
    return "Error: autoresearch-battery-soc.log not found"

# ==========================================
# 4. METRICS MANAGEMENT
# ==========================================
def get_best_metric_from_tsv():
    """Finds best MAE on results.tsv"""
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
# 5. MAIN LOOP (THE AGENT)
# ==========================================
def main_loop():
    program_instructions = read_file("program.md")
    
    # Initializes the agent's memory
    conversation_history = [
        {"role": "system", "content": program_instructions},
        {"role": "user", "content": "Let's start the experiment. Read the current train.py and propose your first structural modification. Respond ONLY with the new complete code inside a ```python block."}
    ]
    
    iteration = 1
    while True:
        print(f"\n{'='*40}\nITERATION {iteration}\n{'='*40}")
        current_train_py = read_file("train.py")
        
        llm_model = "gpt-5.5"

        # 1. Ask the LLM to think and write the new code
        print(llm_model + " is thinking")
        prompt = f"Here is the current code of train.py:\n```python\n{current_train_py}\n```\nWhat is your next move?"
        conversation_history.append({"role": "user", "content": prompt})
        
        # API Call to the LLM
        try:
            response = client.responses.create(
                model=llm_model,
                input=conversation_history
            )
            ai_message = response.output_text

        except Exception as e:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": conversation_history}
                ]
            )
            ai_message = response.choices[0].message.content

        with open("llm_chat_log.txt", "a") as file:
            file.write(extract_llm_text(ai_message))
            file.write("\n\n\n")
            
        conversation_history.append({"role": "assistant", "content": ai_message})
        
        # 2. Extract and save the code, then commit
        new_code = extract_python_code(ai_message)
        write_file("train.py", new_code)
        
        commit_msg = f"Auto-run iteration {iteration}"
        run_bash(f'git commit -am "{commit_msg}"')
        commit_hash = run_bash("git rev-parse --short HEAD").strip()
        
        # 3. Launch the pipeline on Kaggle
        run_log = run_kaggle_pipeline()

        print(run_log)
        
        # 4. Evaluate results by extracting metrics from the log
        test_mae_match = re.search(r"test_mae_percent:\s+([0-9.]+)", run_log)
        vram_match = re.search(r"peak_vram_mb:\s+([0-9.]+)", run_log)
        
        current_mae = float(test_mae_match.group(1)) if test_mae_match else 0.0
        vram_gb = float(vram_match.group(1))/1024 if vram_match else 0.0
        
        best_mae = get_best_metric_from_tsv()
        
        if current_mae == 0.0:
            status = "crash"
            print("Kaggle training has crashed")
            run_bash("git reset HEAD~1 --hard") # Rollback
        elif current_mae < best_mae:
            status = "keep"
            print(f"IMPROVEMENT! New MAE: {current_mae} (previous: {best_mae})")
        else:
            status = "discard"
            print(f"No improvement (MAE: {current_mae})")
            run_bash("git reset HEAD~1 --hard") # Rollback
            
        # 5. Save to TSV
        tsv_line = f"{commit_hash}\t{current_mae:.4f}\t{vram_gb:.1f}\t{status}\tIteration {iteration}\n"
        with open("results.tsv", "a") as f:
            f.write(tsv_line)
            
        # 6. Provide feedback to the LLM for the next loop
        # Pass only the last 1500 characters of the log to save tokens
        feedback = f"Result of the Kaggle run:\n{run_log[-1500:]}\nStatus: {status}."
        if status == "crash":
            feedback += "\nThe code generated an error or Kaggle went OOM. Fix the bug or simplify the architecture."
        elif status == "discard":
            feedback += "\nYour modifications worsened or did not improve the metrics. The code has been rolled back to the previous version. Try another approach."
            
        conversation_history.append({"role": "user", "content": feedback})
        iteration += 1

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nExecution manually interrupted by the user.")