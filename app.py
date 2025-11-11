import os
import sqlite3
from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama
from git import Repo, GitCommandError

# --- 1. CONFIGURATION ---
# Get the absolute path of the directory this script is in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set paths for the model and database
MODEL_PATH = os.path.join(BASE_DIR, "model.gguf")
DATABASE_PATH = os.path.join(BASE_DIR, "ai_memory.db")

# Set how many layers to offload to the GPU (-1 = all).
# Set to 0 to run in CPU-only mode.
N_GPU_LAYERS = 0

# --- 2. INITIALIZE APP & LOAD MODEL ---
# (This part runs only ONCE when you start the server)

print("Starting Flask app...")
# We tell Flask that our 'index.html' (template) is in the same folder
app = Flask(__name__, template_folder=BASE_DIR)

print(f"Loading LLM from: {MODEL_PATH}")
# This loads the model into memory. It can take 30-60 seconds.
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,         # Context window size (how much text it can "remember" at once)
    n_gpu_layers=N_GPU_LAYERS, # Set to 0 for CPU-only
    verbose=True        # Show details in the terminal
)
print("LLM loaded successfully.")

# --- 3. DATABASE SETUP ---
def init_db():
    """Creates the ai_memory.db if it doesn't exist."""
    print(f"Initializing database at: {DATABASE_PATH}")
    try:
        con = sqlite3.connect(DATABASE_PATH)
        cur = con.cursor()
        # Create a table to store past documentation
        # commit_hash: The Git commit ID
        # summary: The AI-generated documentation for that commit
        cur.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                commit_hash TEXT PRIMARY KEY,
                summary TEXT
            )
        ''')
        con.commit()
        con.close()
        print("Database initialized.")
    except Exception as e:
        print(f"Error initializing database: {e}")

# --- 4. FLASK ROUTES (THE WEB SERVER) ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    # This will look for a file named 'index.html' in the same directory
    # It won't work yet, because we haven't made it!
    return render_template('index.html')


@app.route('/api/generate_docs', methods=['POST'])
def api_generate_docs():
    """The main API endpoint that does all the work."""
    print("Received API request...")
    try:
        # 1. Get data from the frontend's JavaScript
        data = request.json
        project_path = data.get('project_path')
        user_prompt = data.get('user_prompt', "Write a clear changelog.") # Default prompt

        if not project_path or not os.path.isdir(project_path):
            return jsonify({'error': 'Invalid or missing project path'}), 400

        # 2. Get Git History
        repo = Repo(project_path)
        latest_commit = repo.head.commit
        commit_hash = latest_commit.hexsha
        commit_message = latest_commit.message.strip()
        
        # Get the diff from the previous commit
        parent_commit = latest_commit.parents[0] if latest_commit.parents else None
        
        diff_text = ""
        if parent_commit:
            diff_text = repo.git.diff(parent_commit, latest_commit)
        else:
            # This is the very first commit. Diff against an "empty tree".
            diff_text = repo.git.diff(repo.git.hash_object('-t', 'tree', '/dev/null'), latest_commit)

        # 3. Get Past "Memory" from DB
        con = sqlite3.connect(DATABASE_PATH)
        con.row_factory = sqlite3.Row # Allows accessing columns by name
        cur = con.cursor()
        
        past_summary = "No previous summary found (this might be the first commit)."
        if parent_commit:
            cur.execute("SELECT summary FROM memory WHERE commit_hash = ?", (parent_commit.hexsha,))
            result = cur.fetchone()
            if result:
                past_summary = result['summary']
        
        # 4. Construct the Final Prompt for the AI
        final_prompt = f"""
        You are an AI assistant documenting a software project.
        The user wants you to follow this instruction: "{user_prompt}"

        Here is the context about the project's history:
        ---[PREVIOUS SUMMARY (commit {parent_commit.hexsha[:7] if parent_commit else 'N/A'})]---
        {past_summary}
        ---[END PREVIOUS SUMMARY]---

        Now, here is the NEW update (commit {commit_hash[:7]}):
        - Developer's Commit Message: "{commit_message}"
        - Raw Code Diff:
        {diff_text}
        ---[END NEW UPDATE]---

        YOUR TASK: Based on all this context, write the new documentation summary for the NEW update (commit {commit_hash[:7]}).
        Follow the user's instruction. Do not include the diff or commit message in your output.
        Only provide the new documentation summary.
        """

        print(f"Generating documentation for commit: {commit_hash[:7]}")
        
        # 5. Call the LLM
        output = llm(
            prompt=final_prompt,
            max_tokens=512,  # Max words to generate
            stop=["---"],    # Stop generating when it sees this
            temperature=0.7, # Makes the output a bit more creative
            echo=False       # Don't repeat the prompt in the output
        )
        
        generated_text = output['choices'][0]['text'].strip()

        # 6. Save New Summary to Memory
        print(f"Saving new summary for {commit_hash} to memory...")
        # Use INSERT OR REPLACE to add or update the entry
        cur.execute("INSERT OR REPLACE INTO memory (commit_hash, summary) VALUES (?, ?)", 
                    (commit_hash, generated_text))
        con.commit()
        con.close()

        # 7. Send Response to Frontend
        print("Done. Sending response to frontend.")
        return jsonify({'documentation': generated_text, 'commit_hash': commit_hash})

    except GitCommandError as e:
        print(f"Git error: {e}")
        return jsonify({'error': f'Git error: {str(e)}'}), 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# --- 5. RUN THE APP ---
if __name__ == '__main__':
    init_db()  # Create the database on first run
    # '127.0.0.1' (localhost) is the most secure default.
    app.run(host='127.0.0.1', port=5000, debug=False)
