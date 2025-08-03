-- Initial database schema for robo-rlhf-multimodal
-- PostgreSQL version

-- Create demonstrations table
CREATE TABLE IF NOT EXISTS demonstrations (
    id SERIAL PRIMARY KEY,
    episode_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    environment VARCHAR(100) NOT NULL,
    task VARCHAR(100) NOT NULL,
    success BOOLEAN DEFAULT FALSE,
    episode_length INTEGER NOT NULL,
    total_reward FLOAT NOT NULL,
    duration_seconds FLOAT NOT NULL,
    data_path TEXT NOT NULL,
    video_path TEXT,
    annotator_id VARCHAR(100),
    device_type VARCHAR(50) NOT NULL,
    observation_modalities JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for demonstrations
CREATE INDEX idx_demo_episode_id ON demonstrations(episode_id);
CREATE INDEX idx_demo_timestamp ON demonstrations(timestamp DESC);
CREATE INDEX idx_demo_success ON demonstrations(success);
CREATE INDEX idx_demo_task_success ON demonstrations(task, success);
CREATE INDEX idx_demo_annotator ON demonstrations(annotator_id);

-- Create preferences table
CREATE TABLE IF NOT EXISTS preferences (
    id SERIAL PRIMARY KEY,
    pair_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    segment_a_demo_id INTEGER REFERENCES demonstrations(id),
    segment_a_start INTEGER NOT NULL,
    segment_a_end INTEGER NOT NULL,
    segment_b_demo_id INTEGER REFERENCES demonstrations(id),
    segment_b_start INTEGER NOT NULL,
    segment_b_end INTEGER NOT NULL,
    annotator_id VARCHAR(100) NOT NULL,
    choice VARCHAR(20) NOT NULL CHECK (choice IN ('a', 'b', 'equal', 'unclear')),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    time_taken_seconds FLOAT NOT NULL,
    session_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for preferences
CREATE INDEX idx_pref_pair_id ON preferences(pair_id);
CREATE INDEX idx_pref_annotator ON preferences(annotator_id);
CREATE INDEX idx_pref_timestamp ON preferences(timestamp DESC);
CREATE INDEX idx_pref_session ON preferences(session_id);
CREATE INDEX idx_pref_annotator_time ON preferences(annotator_id, timestamp DESC);

-- Create training_runs table
CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    model_type VARCHAR(100) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    hyperparameters JSONB NOT NULL,
    num_demonstrations INTEGER NOT NULL,
    num_preferences INTEGER NOT NULL,
    data_split JSONB NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER NOT NULL,
    best_validation_score FLOAT,
    final_test_score FLOAT,
    metrics_history JSONB,
    gpu_hours FLOAT,
    peak_memory_gb FLOAT,
    checkpoint_path TEXT,
    tensorboard_path TEXT,
    wandb_run_id VARCHAR(100),
    git_commit VARCHAR(40),
    environment_info JSONB,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for training_runs
CREATE INDEX idx_run_id ON training_runs(run_id);
CREATE INDEX idx_run_start_time ON training_runs(start_time DESC);
CREATE INDEX idx_run_status ON training_runs(status);
CREATE INDEX idx_run_model_type ON training_runs(model_type);

-- Create model_checkpoints table
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id SERIAL PRIMARY KEY,
    checkpoint_id VARCHAR(100) UNIQUE NOT NULL,
    training_run_id INTEGER REFERENCES training_runs(id),
    epoch INTEGER NOT NULL,
    step INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validation_score FLOAT NOT NULL,
    training_loss FLOAT NOT NULL,
    metrics JSONB,
    file_path TEXT NOT NULL,
    file_size_mb FLOAT NOT NULL,
    is_deployed BOOLEAN DEFAULT FALSE,
    deployment_timestamp TIMESTAMP,
    deployment_environment VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for model_checkpoints
CREATE INDEX idx_checkpoint_id ON model_checkpoints(checkpoint_id);
CREATE INDEX idx_checkpoint_run_score ON model_checkpoints(training_run_id, validation_score DESC);
CREATE INDEX idx_checkpoint_deployed ON model_checkpoints(is_deployed, deployment_timestamp);

-- Create annotation_sessions table for tracking annotation sessions
CREATE TABLE IF NOT EXISTS annotation_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    annotator_id VARCHAR(100) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    num_annotations INTEGER DEFAULT 0,
    total_time_seconds FLOAT DEFAULT 0,
    metadata JSONB
);

-- Create index for annotation_sessions
CREATE INDEX idx_session_annotator ON annotation_sessions(annotator_id);
CREATE INDEX idx_session_time ON annotation_sessions(start_time DESC);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_demonstrations_updated_at 
    BEFORE UPDATE ON demonstrations 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_runs_updated_at 
    BEFORE UPDATE ON training_runs 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create view for demonstration statistics
CREATE OR REPLACE VIEW demonstration_stats AS
SELECT 
    task,
    COUNT(*) as total_demos,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_demos,
    AVG(total_reward) as avg_reward,
    AVG(episode_length) as avg_episode_length,
    AVG(duration_seconds) as avg_duration,
    MIN(timestamp) as first_demo,
    MAX(timestamp) as last_demo
FROM demonstrations
GROUP BY task;

-- Create view for annotator statistics
CREATE OR REPLACE VIEW annotator_stats AS
SELECT 
    annotator_id,
    COUNT(*) as total_annotations,
    AVG(confidence) as avg_confidence,
    AVG(time_taken_seconds) as avg_time_taken,
    COUNT(DISTINCT session_id) as num_sessions,
    MIN(timestamp) as first_annotation,
    MAX(timestamp) as last_annotation
FROM preferences
GROUP BY annotator_id;

-- Create view for training run summary
CREATE OR REPLACE VIEW training_summary AS
SELECT 
    model_type,
    algorithm,
    COUNT(*) as num_runs,
    AVG(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_rate,
    AVG(gpu_hours) as avg_gpu_hours,
    MAX(best_validation_score) as best_score,
    AVG(best_validation_score) as avg_score
FROM training_runs
GROUP BY model_type, algorithm;