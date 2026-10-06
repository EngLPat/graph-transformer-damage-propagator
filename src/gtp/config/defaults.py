"""
Default configuration values for GTP.

These values are used when not overridden by CLI arguments or config file.
"""

DEFAULTS = {
    # ==========================================================================
    # DATASET
    # ==========================================================================
    "dataset_filename": "processed_data_3d_time_prop_1000_steps.pt",
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "interleave_step": 3,  # Time-series interleaved sampling step
    
    # ==========================================================================
    # MODEL ARCHITECTURE
    # ==========================================================================
    "num_layers": 6,
    "hidden_dim": 128,
    "dropout_rate": 0.158,
    "attention_freq": 2,  # global attention every attention_freq layers
    "num_heads": 1,
    "attention_dropout": 0.0,
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================
    "batch_size": 1,
    "epochs": 500,
    "patience": 400,
    "learning_rate": 0.000987,
    "weight_decay": 1e-5,
    "optimizer": "adam",
    "scheduler": "step",
    "scheduler_step": 50,
    "scheduler_gamma": 0.85,
    "scheduler_restart": 0,
    
    # Gradient settings
    "gradient_accumulation_steps": 4,
    "gradient_clip_max_norm": 1.0,
    "gpu_cache_clear_freq": 5,
    
    # Early stopping
    "early_stopping_patience": 1500,
    
    # Checkpointing
    "checkpoint_save_freq": 10,
    "memory_monitor_freq": 20,
    
    # ==========================================================================
    # HIGH-RATE ("SHOCK") REGIME
    # ==========================================================================
    # Nodes whose rate exceeds shock_rate_threshold form the high-rate regime:
    # the rate loss masks them out only for a negative (legacy) rate_cap, and
    # the regime breakdown of test_metrics.txt reports them separately, as the
    # crack-front transition D -> shock_damage_threshold when shock_gt_mode is
    # "damage".
    "shock_gt_mode": "damage",
    "shock_damage_threshold": 0.90,
    "shock_rate_threshold": 100.0,

    # ==========================================================================
    # EARLY TIMESTEP WEIGHTING
    # ==========================================================================
    # Earlier timesteps are weighted more heavily since errors compound
    "early_timestep_weighting_enabled": True,
    "early_timestep_weight_max": 3.0,  # Weight for first timesteps
    "early_timestep_weight_min": 1.0,  # Weight for later timesteps
    "early_timestep_decay_rate": 0.1,  # How fast weight decays (higher = faster decay)
    
    # ==========================================================================
    # LOSS FUNCTION
    # ==========================================================================
    "damage_loss_weight": 1.0,
    "active_node_weight": 1.0,  # weight for active nodes (rate > 0.01) in the loss
    "rate_cap": 10.0,  # clamp of the target rates in the rate loss (see --rate-cap)
    "fixed_high_damage_weight": 3.0,
    "catastrophic_damage_rate_threshold": 100.0,
    "use_unified_decoder": True,
    "log_space_loss_epsilon": 1e-6,
    "damage_weight_increment": 0.1,
    "damage_weight_max": 10.0,
    
    # ==========================================================================
    # TIME ENCODING
    # ==========================================================================
    "total_simulation_time": 1.0,
    
    # ==========================================================================
    # TWO-STAGE HEAD (FI rate + damage rate)
    # ==========================================================================
    "two_stage_enabled": False,      # When True, adds a second decoder head for damage rate
    "dmg_loss_weight": 1.0,          # Weight for the damage-rate loss term
    "dmg_fi_threshold": 1.0,         # FI threshold: only supervise damage rate where FI > this

    # ==========================================================================
    # ATTENTION
    # ==========================================================================
    "local_attention_enabled": True,
    "global_attention_enabled": True,
    "alpha_global": 0.2,  # weight of the global-attention residual
    "time_injection_enabled": True,         # Mid-network (t, dt) injection at layer 3.
                                            # Set False to force attention to carry the
                                            # loading signal.
    "memory_efficient_threshold": 2000,
    "attention_chunk_size": 1000,

    # ==========================================================================
    # DDP SETTINGS
    # ==========================================================================
    "ddp_timeout_minutes": 30,

    # ==========================================================================
    # NORMALIZATION
    # ==========================================================================
    "normalization_epsilon": 1e-6,
    "normalization_max_accumulations": 10**6,
    "normalization_variance_epsilon": 1e-8,
    
    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    "plot_dpi": 300,
    "debug_print_probability": 0.02,
    "loss_debug_probability": 0.01,
    
    # ==========================================================================
    # OUTPUT
    # ==========================================================================
    "results_folder_prefix": "gtp_run",
    
    # ==========================================================================
    # VERBOSITY
    # ==========================================================================
    "verbose": True,
    "verbose_gpu_memory": False,
    "verbose_loss_debug": True,
    "verbose_normalization": True,
    "verbose_data_loading": True,
    "verbose_model_init": True,
    "verbose_training_progress": True,
    "verbose_visualization": True,
    
    # ==========================================================================
    # OPTUNA HYPERPARAMETER SEARCH
    # ==========================================================================
    "optuna_n_trials": 100,
    "optuna_num_layers_min": 4,
    "optuna_num_layers_max": 12,
    "optuna_batch_sizes": [2, 4, 6],
    "optuna_hidden_dims": [32, 64, 96],
    "optuna_dropout_min": 0.1,
    "optuna_dropout_max": 0.4,
    "optuna_attention_freq_min": 3,
    "optuna_attention_freq_max": 8,
    "optuna_opts": ["adam", "rmsprop"],
    "optuna_decay_step_min": 30,
    "optuna_decay_step_max": 60,
    "optuna_decay_rate_min": 0.6,
    "optuna_decay_rate_max": 0.9,
    "optuna_weight_decay_min": 1e-6,
    "optuna_weight_decay_max": 1e-3,
    "optuna_lr_min": 1e-4,
    "optuna_lr_max": 1e-2,
    "optuna_trial_epochs": 50,
}
