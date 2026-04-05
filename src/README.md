# Ticket Monarch — Source Directory

All commands in this README assume you are in the `src/` directory with your virtual environment activated.

## Setup

### 1. Virtual environment

**PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install everything

```bash
pip install -r TicketMonarch/backend/requirements.txt
pip install -r rl_captcha/requirements.txt
pip install -r bots/requirements.txt
```

For the LLM bot:
```bash
pip install browser-use playwright langchain-anthropic
playwright install chromium
```

### 3. Database

```bash
cp TicketMonarch/.env.example TicketMonarch/.env
# Edit TicketMonarch/.env and set MYSQL_PASSWORD
python TicketMonarch/backend/setup_mysql.py
```

### 4. Frontend

```bash
cd TicketMonarch/frontend
npm install
cd ../..
```

---

## Run the App

Open two terminals (activate venv in each):

**PowerShell:**
```powershell
# Terminal 1 — Backend (http://localhost:5000)
$env:RL_ALGORITHM="ppo"
python TicketMonarch/backend/app.py

# Terminal 2 — Frontend (http://localhost:3000)
cd TicketMonarch/frontend
npm run dev
```

**macOS / Linux:**
```bash
# Terminal 1
RL_ALGORITHM=ppo python TicketMonarch/backend/app.py

# Terminal 2
cd TicketMonarch/frontend && npm run dev
```

---

## Run Bots

### Selenium bot
```bash
python bots/selenium_bot.py --runs 5 --type scripted
python bots/selenium_bot.py --runs 5 --type linear
python bots/selenium_bot.py --runs 5 --type stealth
```

### LLM bot

Set API keys first, then enable `DISABLE_HUMAN_SAVE` to prevent LLM sessions from being saved as human data:

**PowerShell:**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:OPENAI_API_KEY="sk-..."
$env:DISABLE_HUMAN_SAVE="1"
python TicketMonarch/backend/app.py   # Restart backend with env var set
```

Then in another terminal:
```bash
python bots/llm_bot.py --runs 3 --provider anthropic
python bots/llm_bot.py --runs 3 --provider openai
```

See [bots/README.md](bots/README.md) for all flags.

---

## Train the RL Agent

### 1. Generate augmented data (one time)
```powershell
python -u -m rl_captcha.scripts.generate_augmented_data --data-dir data/ --n-copies 2 --seed 42 2>&1 | Tee-Object -FilePath logs/generate_augmented.log
```

### 2. Train
```powershell
# PPO without augmentation
python -u -m rl_captcha.scripts.train_ppo --algorithm ppo --data-dir data/ --save-path rl_captcha/agent/checkpoints/ppo_noaug --total-timesteps 500000 2>&1 | Tee-Object -FilePath logs/ppo_noaug_training.log

# PPO with adversarial augmentation
python -u -m rl_captcha.scripts.train_ppo --algorithm ppo --adversarial-augment --data-dir data/ --save-path rl_captcha/agent/checkpoints/ppo_advaug --total-timesteps 500000 2>&1 | Tee-Object -FilePath logs/ppo_advaug_training.log
```

### 3. Evaluate
```powershell
python -u -m rl_captcha.scripts.evaluate_ppo --agent rl_captcha/agent/checkpoints/ppo_noaug --data-dir data/ --episodes 500 --split test 2>&1 | Tee-Object -FilePath logs/eval.log
```

### 4. Plot
```powershell
python -m rl_captcha.scripts.plot_training --log logs/ppo_noaug_training.log --out figures/ppo_noaug
python -m rl_captcha.scripts.plot_eval --log logs/eval.log --out figures/eval
```

See [rl_captcha/README.md](rl_captcha/README.md) for all training/eval commands and flags.

---

## Heatmap Visualization

See [data/README.md](data/README.md).

```bash
python data/gen_heatmap.py --event click --page checkout
```

---

## Project Structure

```
src/
├── TicketMonarch/          # Web app (Flask backend + React frontend)
├── rl_captcha/             # RL agents: PPO, DG, Soft PPO
├── bots/                   # Selenium & LLM bots
├── data/                   # Training data (human/, bot/, bot_augmented/)
└── chrome-extension/       # Telemetry capture extension
```
