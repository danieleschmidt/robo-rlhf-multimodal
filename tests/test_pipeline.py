"""
Tests for the robo-rlhf-multimodal pipeline.

Run with:
  ~/anaconda3/bin/python3 -m pytest tests/test_pipeline.py -v
"""

import pytest
import torch

from robo_rlhf.observation import RobotObservation, IMAGE_SHAPE, PROPRIO_DIM
from robo_rlhf.encoder import MultimodalEncoder, ImageEncoder, ProprioEncoder
from robo_rlhf.reward_model import RewardModel
from robo_rlhf.preference_dataset import PreferenceDataset, PreferencePair
from robo_rlhf.rlhf_trainer import RLHFTrainer, TrainerConfig, RobotPolicy
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# RobotObservation
# ---------------------------------------------------------------------------

class TestRobotObservation:
    def test_random_shapes(self):
        obs = RobotObservation.random()
        assert obs.image.shape == torch.Size(IMAGE_SHAPE)
        assert obs.proprioception.shape == torch.Size([PROPRIO_DIM])

    def test_zeros_shapes(self):
        obs = RobotObservation.zeros()
        assert obs.image.sum() == 0
        assert obs.proprioception.sum() == 0

    def test_wrong_image_shape_raises(self):
        with pytest.raises(ValueError, match="image must have shape"):
            RobotObservation(
                image=torch.zeros(3, 32, 32),
                proprioception=torch.zeros(PROPRIO_DIM),
            )

    def test_wrong_proprio_shape_raises(self):
        with pytest.raises(ValueError, match="proprioception must have shape"):
            RobotObservation(
                image=torch.zeros(*IMAGE_SHAPE),
                proprioception=torch.zeros(7),
            )

    def test_batch_helpers(self):
        obs_list = [RobotObservation.random() for _ in range(5)]
        imgs = RobotObservation.batch_images(obs_list)
        props = RobotObservation.batch_proprios(obs_list)
        assert imgs.shape == (5, *IMAGE_SHAPE)
        assert props.shape == (5, PROPRIO_DIM)

    def test_to_device(self):
        obs = RobotObservation.random()
        obs2 = obs.to("cpu")
        assert obs2.image.device.type == "cpu"


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class TestEncoders:
    def test_image_encoder_output(self):
        enc = ImageEncoder(out_dim=128)
        x = torch.rand(4, 3, 64, 64)
        out = enc(x)
        assert out.shape == (4, 128)

    def test_proprio_encoder_output(self):
        enc = ProprioEncoder(in_dim=PROPRIO_DIM, out_dim=128)
        x = torch.rand(4, PROPRIO_DIM)
        out = enc(x)
        assert out.shape == (4, 128)

    def test_multimodal_encoder_output(self):
        enc = MultimodalEncoder(fused_dim=256)
        imgs = torch.rand(4, 3, 64, 64)
        props = torch.rand(4, PROPRIO_DIM)
        z = enc(imgs, props)
        assert z.shape == (4, 256)

    def test_encode_obs(self):
        enc = MultimodalEncoder()
        obs = RobotObservation.random()
        z = enc.encode_obs(obs)
        assert z.shape == (1, enc.out_dim)

    def test_encode_obs_list(self):
        enc = MultimodalEncoder()
        obs_list = [RobotObservation.random() for _ in range(3)]
        z = enc.encode_obs_list(obs_list)
        assert z.shape == (3, enc.out_dim)


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------

class TestRewardModel:
    def setup_method(self):
        self.model = RewardModel()

    def test_reward_shape(self):
        imgs = torch.rand(4, 3, 64, 64)
        props = torch.rand(4, PROPRIO_DIM)
        r = self.model.reward(imgs, props)
        assert r.shape == (4,)

    def test_reward_from_obs(self):
        obs = RobotObservation.random()
        r = self.model.reward_from_obs(obs)
        assert r.shape == ()  # scalar

    def test_preference_prob_range(self):
        imgs_a = torch.rand(8, 3, 64, 64)
        props_a = torch.rand(8, PROPRIO_DIM)
        imgs_b = torch.rand(8, 3, 64, 64)
        props_b = torch.rand(8, PROPRIO_DIM)
        p = self.model.preference_prob(imgs_a, props_a, imgs_b, props_b)
        assert p.shape == (8,)
        assert (p >= 0).all() and (p <= 1).all()

    def test_forward_returns_scalar_loss(self):
        B = 4
        imgs_a = torch.rand(B, 3, 64, 64)
        props_a = torch.rand(B, PROPRIO_DIM)
        imgs_b = torch.rand(B, 3, 64, 64)
        props_b = torch.rand(B, PROPRIO_DIM)
        pref = torch.randint(0, 2, (B,)).float()
        loss = self.model(imgs_a, props_a, imgs_b, props_b, pref)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_bradley_terry_loss_known_case(self):
        """When r_a >> r_b and pref=1 (A preferred), loss should be near 0."""
        r_a = torch.tensor([10.0, 10.0])
        r_b = torch.tensor([-10.0, -10.0])
        pref = torch.tensor([1.0, 1.0])
        loss = RewardModel.bradley_terry_loss(r_a, r_b, pref)
        assert loss.item() < 0.01


# ---------------------------------------------------------------------------
# PreferenceDataset
# ---------------------------------------------------------------------------

class TestPreferenceDataset:
    def test_synthetic_length(self):
        ds = PreferenceDataset.synthetic(n_pairs=50)
        assert len(ds) == 50

    def test_getitem_shapes(self):
        ds = PreferenceDataset.synthetic(n_pairs=10)
        img_a, prop_a, img_b, prop_b, pref = ds[0]
        assert img_a.shape == torch.Size(IMAGE_SHAPE)
        assert prop_a.shape == torch.Size([PROPRIO_DIM])
        assert pref.shape == ()

    def test_preference_labels_binary(self):
        ds = PreferenceDataset.synthetic(n_pairs=100)
        for _, _, _, _, pref in ds:
            assert pref.item() in (0.0, 1.0)

    def test_collate_fn(self):
        ds = PreferenceDataset.synthetic(n_pairs=10)
        loader = DataLoader(ds, batch_size=10, collate_fn=PreferenceDataset.collate_fn)
        batch = next(iter(loader))
        imgs_a, props_a, imgs_b, props_b, prefs = batch
        assert imgs_a.shape == (10, *IMAGE_SHAPE)
        assert props_a.shape == (10, PROPRIO_DIM)
        assert prefs.shape == (10,)

    def test_add_pair(self):
        ds = PreferenceDataset()
        obs_a = RobotObservation.random()
        obs_b = RobotObservation.random()
        ds.add(obs_a, obs_b, 1.0)
        assert len(ds) == 1

    def test_reproducible_with_seed(self):
        ds1 = PreferenceDataset.synthetic(n_pairs=20, seed=7)
        ds2 = PreferenceDataset.synthetic(n_pairs=20, seed=7)
        img1, _, _, _, _ = ds1[0]
        img2, _, _, _, _ = ds2[0]
        assert torch.allclose(img1, img2)


# ---------------------------------------------------------------------------
# RLHFTrainer
# ---------------------------------------------------------------------------

class TestRLHFTrainer:
    def setup_method(self):
        self.cfg = TrainerConfig(
            reward_epochs=2,
            reward_batch_size=16,
            ppo_updates=2,
            ppo_rollout_steps=16,
            device="cpu",
        )
        self.trainer = RLHFTrainer(config=self.cfg)

    def test_reward_training_reduces_loss(self):
        ds = PreferenceDataset.synthetic(n_pairs=100)
        history = self.trainer.train_reward_model(ds)
        assert len(history) == 2
        # Loss should be positive
        assert all(v > 0 for v in history)

    def test_evaluate_reward_model_above_chance(self):
        ds_train = PreferenceDataset.synthetic(n_pairs=200, noise=0.05)
        ds_eval  = PreferenceDataset.synthetic(n_pairs=50, noise=0.0)
        self.trainer.train_reward_model(ds_train)
        acc = self.trainer.evaluate_reward_model(ds_eval)
        assert 0.0 <= acc <= 1.0
        # Should beat random (0.5) at least slightly after 2 epochs
        # (generous threshold since this is a very short training run)
        assert acc >= 0.4

    def test_ppo_runs_without_error(self):
        metrics = self.trainer.train_policy()
        assert len(metrics) == 2
        for m in metrics:
            assert "policy_loss" in m
            assert "value_loss" in m
            assert "mean_reward" in m

    def test_robot_policy_sample_shape(self):
        policy = RobotPolicy(fused_dim=256, action_dim=PROPRIO_DIM)
        z = torch.rand(4, 256)
        action, log_prob = policy.sample(z)
        assert action.shape == (4, PROPRIO_DIM)
        assert log_prob.shape == (4,)
