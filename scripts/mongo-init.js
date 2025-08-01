// MongoDB initialization script for Robo-RLHF-Multimodal
// This script sets up the initial database structure and indexes

// Switch to the robo_rlhf database
db = db.getSiblingDB('robo_rlhf');

// Create collections with validation schemas
db.createCollection('demonstrations', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["episode_id", "task", "observations", "actions", "created_at"],
      properties: {
        episode_id: { bsonType: "string" },
        task: { bsonType: "string" },
        observations: { bsonType: "array" },
        actions: { bsonType: "array" },
        rewards: { bsonType: "array" },
        metadata: { bsonType: "object" },
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" }
      }
    }
  }
});

db.createCollection('preferences', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["pair_id", "trajectory_1", "trajectory_2", "preference", "annotator_id", "created_at"],
      properties: {
        pair_id: { bsonType: "string" },
        trajectory_1: { bsonType: "object" },
        trajectory_2: { bsonType: "object" },
        preference: { bsonType: "int", minimum: -1, maximum: 1 },
        confidence: { bsonType: "double", minimum: 0, maximum: 1 },
        annotator_id: { bsonType: "string" },
        annotation_time: { bsonType: "double" },
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" }
      }
    }
  }
});

db.createCollection('models', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["model_id", "model_type", "checkpoint_path", "created_at"],
      properties: {
        model_id: { bsonType: "string" },
        model_type: { bsonType: "string", enum: ["policy", "reward", "encoder"] },
        checkpoint_path: { bsonType: "string" },
        config: { bsonType: "object" },
        metrics: { bsonType: "object" },
        version: { bsonType: "string" },
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" }
      }
    }
  }
});

db.createCollection('experiments', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["experiment_id", "name", "config", "status", "created_at"],
      properties: {
        experiment_id: { bsonType: "string" },
        name: { bsonType: "string" },
        description: { bsonType: "string" },
        config: { bsonType: "object" },
        status: { bsonType: "string", enum: ["running", "completed", "failed", "paused"] },
        metrics: { bsonType: "object" },
        results: { bsonType: "object" },
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" },
        completed_at: { bsonType: "date" }
      }
    }
  }
});

// Create indexes for better query performance
db.demonstrations.createIndex({ "episode_id": 1 }, { unique: true });
db.demonstrations.createIndex({ "task": 1 });
db.demonstrations.createIndex({ "created_at": -1 });
db.demonstrations.createIndex({ "metadata.success": 1 });

db.preferences.createIndex({ "pair_id": 1 }, { unique: true });
db.preferences.createIndex({ "annotator_id": 1 });
db.preferences.createIndex({ "created_at": -1 });
db.preferences.createIndex({ "preference": 1 });

db.models.createIndex({ "model_id": 1 }, { unique: true });
db.models.createIndex({ "model_type": 1 });
db.models.createIndex({ "created_at": -1 });
db.models.createIndex({ "version": 1 });

db.experiments.createIndex({ "experiment_id": 1 }, { unique: true });
db.experiments.createIndex({ "name": 1 });
db.experiments.createIndex({ "status": 1 });
db.experiments.createIndex({ "created_at": -1 });

// Create a user for the application
db.createUser({
  user: "robo_app",
  pwd: "robo_password",
  roles: [
    { role: "readWrite", db: "robo_rlhf" }
  ]
});

print("MongoDB initialization completed for Robo-RLHF-Multimodal");
print("Created collections: demonstrations, preferences, models, experiments");
print("Created indexes for optimal query performance");
print("Created application user: robo_app");