# Generate Telemetry Heatmap

### Setup

Assumes you are in a venv already.

```bash
pip install -r requirements.txt
```

Script expects three data folders in the same directory.\
Each folder should have json data files following the structure defined in /bot and /human READMEs

Folders expected:
- human
- bot
- bot_augmented

### Usage

2 Flags\
--event
- mouse (Mouse movement events)
- click (Mouse click events)

--page
- home (Home page)
- seat (Seat selection page)
- checkout (Checkout page)

```bash
Example:
python gen_heatmap.py --event click --page checkout
```

The will display a heatmap for the event selected on the page selected.

### Data

Heatmap data is printed in terminal for each map.

- Mean (X, Y) pixel position of data
- Total data points used for map
- Spatial entropy of map (Higher = more spread out | Lower = more concentrated)
- Occupied bins out of the total bin number (bins create 60 x 60 grid = 3600 total bins)
- Peak concentration (% of bin with highest % of events)
- X-axis variance (tracks horizontal spread)
- Unique X-axis positions
