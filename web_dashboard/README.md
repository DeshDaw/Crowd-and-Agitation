# Crowd Surveillance Dashboard

A modern, research-grade web dashboard for Abnormal Crowd Motion Detection with agitation index analysis.

## Architecture

The dashboard consists of:
- **Backend**: FastAPI application that wraps the existing crowd_project pipeline
- **Frontend**: React + TypeScript + Tailwind CSS dashboard
- **Storage**: Isolated run directories with JSON status files

## Directory Structure

```
web_dashboard/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── models.py            # Pydantic models
│   │   ├── routers/
│   │   │   ├── runs.py          # Run CRUD endpoints
│   │   │   ├── files.py         # File download endpoints
│   │   │   └── health.py        # Health check
│   │   └── services/
│   │       ├── runner.py        # Background processing
│   │       └── storage.py       # File storage utilities
│   ├── requirements.txt
│   └── run.py                   # Entry point
│
└── frontend/
    ├── src/
    │   ├── api/                 # API client
    │   ├── components/
    │   │   ├── ui/              # Reusable UI components
    │   │   ├── upload/          # Upload components
    │   │   ├── progress/        # Progress monitoring
    │   │   ├── results/         # Results display
    │   │   └── viewer/          # Image viewer
    │   ├── hooks/               # Custom React hooks
    │   ├── pages/               # Page components
    │   ├── store/               # Zustand state management
    │   └── types/               # TypeScript types
    ├── package.json
    ├── vite.config.ts
    └── tailwind.config.js
```

## Setup Instructions

### Backend Setup

1. Install FastAPI dependencies:
```bash
cd web_dashboard/backend
pip install -r requirements.txt
```

2. Start the API server:
```bash
python run.py
```

The API will be available at `http://localhost:8000`.

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd web_dashboard/frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The dashboard will be available at `http://localhost:5173`.

## Usage

### Creating a New Run

1. Click "New Run" on the dashboard
2. Upload images (jpg/png/webp) or a single video (mp4/avi/mov)
3. Configure processing parameters:
   - Device: CPU or CUDA
   - Confidence thresholds
   - Tracking parameters
   - Density classification thresholds
   - Output options
4. Click "Start Processing"

### Monitoring Progress

- Real-time progress bar with frame count
- Current stage display (inference, classification, event detection)
- Per-stage timing information
- ETA estimation

### Viewing Results

- **Summary Cards**: Aggregate statistics (avg crowd size, peak density, etc.)
- **Classification Chart**: Distribution of crowd classifications
- **Metrics Table**: Per-frame detailed metrics with filtering
- **Events Timeline**: Escalation events with quick links to frames
- **Plots**: Crowd density and agitation trend charts
- **Frame Gallery**: Grid of annotated frames with event highlighting
- **Downloads**: All output files (JSON, PNG, DB)

## API Endpoints

### Runs
- `POST /api/runs` - Create new run
- `POST /api/runs/{run_id}/upload` - Upload files
- `POST /api/runs/{run_id}/start` - Start processing
- `GET /api/runs/{run_id}/status` - Get run status
- `GET /api/runs/{run_id}/summary` - Get summary JSON
- `GET /api/runs/{run_id}/metrics` - Get metrics JSON
- `GET /api/runs/{run_id}/events` - Get events JSON
- `GET /api/runs/{run_id}/files` - List output files
- `POST /api/runs/{run_id}/cancel` - Cancel run
- `DELETE /api/runs/{run_id}` - Delete run

### Artifacts
- `GET /api/runs/{run_id}/artifacts/summary.json`
- `GET /api/runs/{run_id}/artifacts/metrics.json`
- `GET /api/runs/{run_id}/artifacts/events.json`
- `GET /api/runs/{run_id}/artifacts/database.db`
- `GET /api/runs/{run_id}/artifacts/density_plot.png`
- `GET /api/runs/{run_id}/artifacts/agitation_plot.png`
- `GET /api/runs/{run_id}/artifacts/annotated/{filename}`
- `GET /api/runs/{run_id}/artifacts/heatmaps/{filename}`
- `GET /api/runs/{run_id}/artifacts/escalation/{filename}`

### Health
- `GET /api/health` - Health check

## Compatibility

The dashboard is fully compatible with the existing CLI:
- The same `FrameProcessor` class is used
- Config values can be overridden per-run
- Runs are isolated in separate directories
- CLI can still be used independently

## Configuration

Default configuration is inherited from `crowd_project/config.py`. Each run can override:
- `device`: cpu/cuda
- `confidence_threshold`: 0.0-1.0
- `pose_confidence_threshold`: 0.0-1.0
- `max_inference_width`: 320-2048
- `tracker_iou_threshold`: 0.0-1.0
- `tracker_max_lost`: 1-100
- `density_low_sigma`: 0.1-5.0
- `density_high_sigma`: 0.1-10.0
- `agitation_threshold_sigma`: 0.0-5.0
- `video_extract_fps`: 1-60 or null
- `save_annotated`: boolean
- `save_heatmaps`: boolean
- `generate_plots`: boolean
- `save_database`: boolean

## Development

### Backend Development

The backend uses FastAPI with:
- Background processing via threading
- Pydantic for request/response validation
- Isolated run directories with JSON status files
- Direct import of crowd_project modules

### Frontend Development

The frontend uses:
- React 18 with TypeScript
- TanStack Query for data fetching
- Zustand for state management
- Tailwind CSS for styling
- Lucide React for icons
- React Router for navigation
- React Dropzone for file uploads

## License

Research prototype - government/research use.
