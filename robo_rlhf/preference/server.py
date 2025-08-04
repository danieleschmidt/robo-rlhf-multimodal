"""
Web server for collecting human preferences on robot trajectories.
"""

import argparse
import json
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np

from robo_rlhf.preference.models import PreferencePair, PreferenceChoice


class PreferenceAnnotation(BaseModel):
    """Model for preference annotation."""
    pair_id: str
    choice: str  # "A", "B", or "equal"
    confidence: float  # 0.0 to 1.0
    annotator: str
    timestamp: str
    comments: Optional[str] = None


class PreferenceServer:
    """
    Web server for human preference collection.
    
    Serves a web interface where annotators can compare robot trajectories
    and provide preference feedback.
    """
    
    def __init__(
        self,
        pairs: List[PreferencePair],
        port: int = 8080,
        annotators: Optional[List[str]] = None,
        output_file: str = "preferences.json"
    ):
        """
        Initialize preference server.
        
        Args:
            pairs: List of preference pairs to annotate
            port: Server port
            annotators: List of annotator names
            output_file: Output file for collected preferences
        """
        self.pairs = pairs
        self.port = port
        self.annotators = annotators or ["annotator1"]
        self.output_file = Path(output_file)
        self.annotations: List[PreferenceAnnotation] = []
        
        # Create FastAPI app
        self.app = FastAPI(title="Robo-RLHF Preference Collection")
        self._setup_routes()
        
        # Load existing annotations if available
        self._load_existing_annotations()
        
        print(f"Preference server initialized with {len(pairs)} pairs")
        print(f"Annotators: {', '.join(self.annotators)}")
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Serve the main annotation interface."""
            return self._generate_html_interface()
        
        @self.app.get("/api/pairs")
        async def get_pairs():
            """Get all preference pairs."""
            pairs_data = []
            for i, pair in enumerate(self.pairs):
                pairs_data.append({
                    "id": pair.pair_id,
                    "index": i,
                    "segment_a": self._serialize_segment(pair.segment_a),
                    "segment_b": self._serialize_segment(pair.segment_b),
                    "metadata": pair.metadata
                })
            return pairs_data
        
        @self.app.get("/api/pair/{pair_id}")
        async def get_pair(pair_id: str):
            """Get specific preference pair."""
            pair = self._find_pair(pair_id)
            if not pair:
                raise HTTPException(status_code=404, detail="Pair not found")
            
            return {
                "id": pair.pair_id,
                "segment_a": self._serialize_segment(pair.segment_a),
                "segment_b": self._serialize_segment(pair.segment_b),
                "metadata": pair.metadata
            }
        
        @self.app.post("/api/annotate")
        async def submit_annotation(
            pair_id: str = Form(...),
            choice: str = Form(...),
            confidence: float = Form(...),
            annotator: str = Form(...),
            comments: Optional[str] = Form(None)
        ):
            """Submit preference annotation."""
            # Validate inputs
            if choice not in ["A", "B", "equal"]:
                raise HTTPException(status_code=400, detail="Invalid choice")
            
            if not 0.0 <= confidence <= 1.0:
                raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1")
            
            if annotator not in self.annotators:
                raise HTTPException(status_code=400, detail="Invalid annotator")
            
            # Find the pair
            pair = self._find_pair(pair_id)
            if not pair:
                raise HTTPException(status_code=404, detail="Pair not found")
            
            # Create annotation
            annotation = PreferenceAnnotation(
                pair_id=pair_id,
                choice=choice,
                confidence=confidence,
                annotator=annotator,
                timestamp=datetime.now().isoformat(),
                comments=comments
            )
            
            self.annotations.append(annotation)
            
            # Update the pair with the preference
            preference_choice = {
                "A": PreferenceChoice.SEGMENT_A,
                "B": PreferenceChoice.SEGMENT_B,
                "equal": PreferenceChoice.EQUAL
            }[choice]
            
            pair.add_preference(annotator, preference_choice, confidence)
            
            # Save annotations
            self._save_annotations()
            
            return {"status": "success", "message": "Annotation saved"}
        
        @self.app.get("/api/stats")
        async def get_stats():
            """Get annotation statistics."""
            total_pairs = len(self.pairs)
            annotated_pairs = len(set(ann.pair_id for ann in self.annotations))
            
            # Per-annotator stats
            annotator_stats = {}
            for annotator in self.annotators:
                ann_count = sum(1 for ann in self.annotations if ann.annotator == annotator)
                annotator_stats[annotator] = ann_count
            
            # Agreement analysis
            agreement_score = self._calculate_agreement()
            
            return {
                "total_pairs": total_pairs,
                "annotated_pairs": annotated_pairs,
                "completion_rate": annotated_pairs / total_pairs if total_pairs > 0 else 0,
                "total_annotations": len(self.annotations),
                "annotator_stats": annotator_stats,
                "agreement_score": agreement_score
            }
        
        @self.app.get("/api/export")
        async def export_preferences():
            """Export all collected preferences."""
            # Convert pairs to serializable format
            export_data = []
            for pair in self.pairs:
                if pair.preferences:
                    export_data.append({
                        "pair_id": pair.pair_id,
                        "segment_a": self._serialize_segment(pair.segment_a),
                        "segment_b": self._serialize_segment(pair.segment_b),
                        "preferences": [
                            {
                                "annotator": annotator,
                                "choice": choice.value,
                                "confidence": conf
                            }
                            for annotator, (choice, conf) in pair.preferences.items()
                        ],
                        "consensus": pair.get_consensus().value if pair.get_consensus() else None,
                        "agreement_score": pair.get_agreement_score(),
                        "metadata": pair.metadata
                    })
            
            return export_data
    
    def _generate_html_interface(self) -> str:
        """Generate HTML interface for preference annotation."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Robo-RLHF Preference Collection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .comparison {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }
        .segment {
            flex: 1;
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .segment.selected {
            border-color: #007bff;
            background-color: #f0f8ff;
        }
        .segment h3 {
            margin-top: 0;
            color: #333;
        }
        .trajectory-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            text-align: left;
        }
        .controls {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .button-group {
            margin: 10px;
        }
        button {
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn-primary {
            background: #007bff;
            color: white;
        }
        .btn-success {
            background: #28a745;
            color: white;
        }
        .btn-warning {
            background: #ffc107;
            color: black;
        }
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        button:hover {
            opacity: 0.8;
        }
        .slider-container {
            margin: 20px 0;
        }
        .slider {
            width: 300px;
            margin: 10px;
        }
        .stats {
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .progress-bar {
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
        }
        .progress-fill {
            background: #28a745;
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Robo-RLHF Preference Collection</h1>
            <p>Compare robot trajectories and indicate which one you prefer</p>
        </div>
        
        <div id="stats-panel" class="stats">
            <h3>Progress</h3>
            <div class="progress-bar">
                <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
            </div>
            <p id="progress-text">Loading...</p>
        </div>
        
        <div id="comparison-panel" class="comparison">
            <div id="segment-a" class="segment">
                <h3>Trajectory A</h3>
                <div class="trajectory-info">
                    <p><strong>Episode:</strong> <span id="episode-a">-</span></p>
                    <p><strong>Duration:</strong> <span id="duration-a">-</span> steps</p>
                    <p><strong>Success:</strong> <span id="success-a">-</span></p>
                    <p><strong>Actions:</strong> <span id="actions-a">-</span></p>
                </div>
            </div>
            
            <div id="segment-b" class="segment">
                <h3>Trajectory B</h3>
                <div class="trajectory-info">
                    <p><strong>Episode:</strong> <span id="episode-b">-</span></p>
                    <p><strong>Duration:</strong> <span id="duration-b">-</span> steps</p>
                    <p><strong>Success:</strong> <span id="success-b">-</span></p>
                    <p><strong>Actions:</strong> <span id="actions-b">-</span></p>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <div class="button-group">
                <h3>Which trajectory do you prefer?</h3>
                <button id="prefer-a" class="btn-primary" onclick="selectPreference('A')">
                    🅰️ Prefer A
                </button>
                <button id="prefer-equal" class="btn-warning" onclick="selectPreference('equal')">
                    ⚖️ Equal
                </button>
                <button id="prefer-b" class="btn-primary" onclick="selectPreference('B')">
                    🅱️ Prefer B
                </button>
            </div>
            
            <div class="slider-container">
                <label for="confidence">Confidence Level:</label><br>
                <input type="range" id="confidence" class="slider" min="0" max="100" value="50">
                <span id="confidence-value">50%</span>
            </div>
            
            <div>
                <label for="annotator">Annotator:</label>
                <select id="annotator">""" + "".join([f'<option value="{ann}">{ann}</option>' for ann in self.annotators]) + """</select>
            </div>
            
            <div style="margin-top: 20px;">
                <textarea id="comments" placeholder="Optional comments..." rows="3" cols="50"></textarea>
            </div>
            
            <div class="button-group" style="margin-top: 20px;">
                <button id="submit-btn" class="btn-success" onclick="submitAnnotation()" disabled>
                    ✅ Submit Annotation
                </button>
                <button class="btn-secondary" onclick="skipPair()">
                    ⏭️ Skip
                </button>
                <button class="btn-secondary" onclick="exportResults()">
                    📥 Export Results
                </button>
            </div>
        </div>
    </div>

    <script>
        let currentPairIndex = 0;
        let pairs = [];
        let selectedChoice = null;
        
        // Load pairs and initialize
        async function initialize() {
            try {
                const response = await fetch('/api/pairs');
                pairs = await response.json();
                updateStats();
                loadCurrentPair();
            } catch (error) {
                console.error('Failed to load pairs:', error);
                alert('Failed to load preference pairs');
            }
        }
        
        function loadCurrentPair() {
            if (currentPairIndex >= pairs.length) {
                alert('All pairs completed! 🎉');
                return;
            }
            
            const pair = pairs[currentPairIndex];
            
            // Update segment A
            document.getElementById('episode-a').textContent = pair.segment_a.episode_id;
            document.getElementById('duration-a').textContent = pair.segment_a.actions.length;
            document.getElementById('success-a').textContent = pair.segment_a.metadata?.success ? 'Yes' : 'No';
            document.getElementById('actions-a').textContent = `[${pair.segment_a.actions.slice(0, 3).map(a => a.toFixed(2)).join(', ')}...]`;
            
            // Update segment B
            document.getElementById('episode-b').textContent = pair.segment_b.episode_id;
            document.getElementById('duration-b').textContent = pair.segment_b.actions.length;
            document.getElementById('success-b').textContent = pair.segment_b.metadata?.success ? 'Yes' : 'No';
            document.getElementById('actions-b').textContent = `[${pair.segment_b.actions.slice(0, 3).map(a => a.toFixed(2)).join(', ')}...]`;
            
            // Reset selection
            resetSelection();
        }
        
        function selectPreference(choice) {
            selectedChoice = choice;
            
            // Update UI
            document.getElementById('segment-a').classList.toggle('selected', choice === 'A');
            document.getElementById('segment-b').classList.toggle('selected', choice === 'B');
            document.getElementById('submit-btn').disabled = false;
            
            // Update buttons
            document.getElementById('prefer-a').style.background = choice === 'A' ? '#0056b3' : '#007bff';
            document.getElementById('prefer-b').style.background = choice === 'B' ? '#0056b3' : '#007bff';
            document.getElementById('prefer-equal').style.background = choice === 'equal' ? '#e69500' : '#ffc107';
        }
        
        function resetSelection() {
            selectedChoice = null;
            document.getElementById('segment-a').classList.remove('selected');
            document.getElementById('segment-b').classList.remove('selected');
            document.getElementById('submit-btn').disabled = true;
            document.getElementById('confidence').value = 50;
            document.getElementById('confidence-value').textContent = '50%';
            document.getElementById('comments').value = '';
            
            // Reset button colors
            document.getElementById('prefer-a').style.background = '#007bff';
            document.getElementById('prefer-b').style.background = '#007bff';
            document.getElementById('prefer-equal').style.background = '#ffc107';
        }
        
        async function submitAnnotation() {
            if (!selectedChoice) {
                alert('Please select a preference first');
                return;
            }
            
            const pair = pairs[currentPairIndex];
            const confidence = parseInt(document.getElementById('confidence').value) / 100.0;
            const annotator = document.getElementById('annotator').value;
            const comments = document.getElementById('comments').value;
            
            const formData = new FormData();
            formData.append('pair_id', pair.id);
            formData.append('choice', selectedChoice);
            formData.append('confidence', confidence.toString());
            formData.append('annotator', annotator);
            if (comments) formData.append('comments', comments);
            
            try {
                const response = await fetch('/api/annotate', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    currentPairIndex++;
                    updateStats();
                    loadCurrentPair();
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.detail}`);
                }
            } catch (error) {
                console.error('Submission failed:', error);
                alert('Failed to submit annotation');
            }
        }
        
        function skipPair() {
            currentPairIndex++;
            updateStats();
            loadCurrentPair();
        }
        
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                const progress = (currentPairIndex / pairs.length) * 100;
                document.getElementById('progress-fill').style.width = `${progress}%`;
                document.getElementById('progress-text').textContent = 
                    `${currentPairIndex}/${pairs.length} pairs completed (${progress.toFixed(1)}%) - ${stats.total_annotations} total annotations`;
            } catch (error) {
                console.error('Failed to update stats:', error);
            }
        }
        
        async function exportResults() {
            try {
                const response = await fetch('/api/export');
                const data = await response.json();
                
                const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'preferences.json';
                a.click();
                URL.revokeObjectURL(url);
            } catch (error) {
                console.error('Export failed:', error);
                alert('Failed to export results');
            }
        }
        
        // Update confidence display
        document.getElementById('confidence').oninput = function() {
            document.getElementById('confidence-value').textContent = this.value + '%';
        };
        
        // Initialize on load
        initialize();
    </script>
</body>
</html>
        """
        return html
    
    def _serialize_segment(self, segment) -> Dict[str, Any]:
        """Convert segment to JSON-serializable format."""
        return {
            "episode_id": segment.episode_id,
            "start_frame": segment.start_frame,
            "end_frame": segment.end_frame,
            "actions": segment.actions.tolist() if hasattr(segment.actions, 'tolist') else segment.actions,
            "metadata": segment.metadata or {}
        }
    
    def _find_pair(self, pair_id: str) -> Optional[PreferencePair]:
        """Find preference pair by ID."""
        for pair in self.pairs:
            if pair.pair_id == pair_id:
                return pair
        return None
    
    def _calculate_agreement(self) -> float:
        """Calculate inter-annotator agreement score."""
        if len(self.annotators) < 2:
            return 1.0
        
        # Group annotations by pair
        pair_annotations = {}
        for ann in self.annotations:
            if ann.pair_id not in pair_annotations:
                pair_annotations[ann.pair_id] = []
            pair_annotations[ann.pair_id].append(ann)
        
        # Calculate agreement for pairs with multiple annotations
        agreements = []
        for pair_id, anns in pair_annotations.items():
            if len(anns) >= 2:
                # Simple agreement: all annotators chose the same option
                choices = [ann.choice for ann in anns]
                if len(set(choices)) == 1:
                    agreements.append(1.0)
                else:
                    agreements.append(0.0)
        
        return sum(agreements) / len(agreements) if agreements else 0.0
    
    def _load_existing_annotations(self) -> None:
        """Load existing annotations from file."""
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                
                self.annotations = [
                    PreferenceAnnotation(**ann) for ann in data.get('annotations', [])
                ]
                
                # Restore preferences to pairs
                for ann in self.annotations:
                    pair = self._find_pair(ann.pair_id)
                    if pair:
                        choice_map = {
                            "A": PreferenceChoice.SEGMENT_A,
                            "B": PreferenceChoice.SEGMENT_B,
                            "equal": PreferenceChoice.EQUAL
                        }
                        pair.add_preference(
                            ann.annotator,
                            choice_map[ann.choice],
                            ann.confidence
                        )
                
                print(f"Loaded {len(self.annotations)} existing annotations")
            except Exception as e:
                print(f"Warning: Could not load existing annotations: {e}")
    
    def _save_annotations(self) -> None:
        """Save annotations to file."""
        try:
            data = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "total_pairs": len(self.pairs),
                    "annotators": self.annotators
                },
                "annotations": [ann.dict() for ann in self.annotations]
            }
            
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            print(f"Warning: Could not save annotations: {e}")
    
    def run(self) -> None:
        """Start the preference collection server."""
        print(f"Starting preference collection server on port {self.port}")
        print(f"Open http://localhost:{self.port} in your browser")
        
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )


def main(args: argparse.Namespace) -> None:
    """Main server function."""
    # Load preference pairs
    with open(args.pairs, 'rb') as f:
        pairs = pickle.load(f)
    
    # Create and run server
    server = PreferenceServer(
        pairs=pairs,
        port=args.port,
        annotators=args.annotators,
        output_file=args.output
    )
    
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch preference collection server")
    parser.add_argument("--pairs", required=True, help="Path to preference pairs file")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--output", default="data/preferences.json", help="Output file")
    parser.add_argument("--annotators", nargs="+", help="List of annotator names")
    
    args = parser.parse_args()
    main(args)