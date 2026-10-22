# Cost contract

{
  "canonical_idle_power_intentionally_preserved": true,
  "cost": {
    "beta": 20.0,
    "program_type": "RSR",
    "psi1": 1.0,
    "psi2": 10.0,
    "qos_constraint": 0.1,
    "rho": 2.0,
    "source_file": "configs/optimization/simulated_annealing/SA_train_low_util_RSR_1000server.ini",
    "tracking_error_constraint": 0.3
  },
  "cost_sha256": "f20f3f78f53cca79d29c8b077747df86d9a23dc5bf19a5870142154c9b0567c4",
  "gradient_sha256": "b22d5b83468bcd1ca0fb1a6a07c14edc8d6d9e478a2706c710bfbc21f1beb803",
  "node_count_control": true,
  "policy": "peacsim.aqa_runtimepolicy.AQARuntimePolicy",
  "templates": [
    {
      "ISO": {
        "iso_file_path": "../../data/1-17-2025_traditional.csv",
        "iso_signal_granularity": "2",
        "iso_signal_start_hour": "16",
        "normalize_iso_signal": "True",
        "randomize_iso_start": "False"
      },
      "N": 250,
      "U": 0.6,
      "generated": "configs/foundation_v1/experiments/exp_N250_U060.ini",
      "idle_watts": 120.0,
      "sha256": "610050ec146bf83fe1b59ddc31960b3166dc4f11cbcaa42b8cea49bbe18fd4cb",
      "source": "configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_250.ini",
      "source_sha256": "2850ce43e8c57a52939949e3517dd2f55ca7d6f2c636802071e46f8f78d6f521"
    },
    {
      "ISO": {
        "iso_file_path": "../../data/1-17-2025_traditional.csv",
        "iso_signal_granularity": "2",
        "iso_signal_start_hour": "16",
        "normalize_iso_signal": "True",
        "randomize_iso_start": "False"
      },
      "N": 250,
      "U": 0.8,
      "generated": "configs/foundation_v1/experiments/exp_N250_U080.ini",
      "idle_watts": 120.0,
      "sha256": "b278aae3d0921d676128a341d4d3808f4a2296259d460531d7f49c05f8289790",
      "source": "configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_250.ini",
      "source_sha256": "2850ce43e8c57a52939949e3517dd2f55ca7d6f2c636802071e46f8f78d6f521"
    },
    {
      "ISO": {
        "iso_file_path": "../../data/1-17-2025_traditional.csv",
        "iso_signal_granularity": "2",
        "iso_signal_start_hour": "16",
        "normalize_iso_signal": "True",
        "randomize_iso_start": "False"
      },
      "N": 1000,
      "U": 0.6,
      "generated": "configs/foundation_v1/experiments/exp_N1000_U060.ini",
      "idle_watts": 120.0,
      "sha256": "e6363151206af4d485e8c1b21f6a3ac6d4cf778f3dc378f5dc5eb47f528818ba",
      "source": "configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_1000.ini",
      "source_sha256": "e358c47a2fb3f33d6480ef8175d8373117b02916a2c2215e36dea85531b62a7e"
    },
    {
      "ISO": {
        "iso_file_path": "../../data/1-17-2025_traditional.csv",
        "iso_signal_granularity": "2",
        "iso_signal_start_hour": "16",
        "normalize_iso_signal": "True",
        "randomize_iso_start": "False"
      },
      "N": 1000,
      "U": 0.8,
      "generated": "configs/foundation_v1/experiments/exp_N1000_U080.ini",
      "idle_watts": 120.0,
      "sha256": "507007e321dee86936c833fff18ac34992528fd9ef51e6f55e9059a5051926b7",
      "source": "configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_1000.ini",
      "source_sha256": "e358c47a2fb3f33d6480ef8175d8373117b02916a2c2215e36dea85531b62a7e"
    }
  ]
}

Only cost_function/dr_program constants are imported from SA. No initialization/annealing/boundary/step_size settings enter the planner or execution method. Generated gradient config is the existing gradient base with seven canonical cost/DR overrides;no gradient optimizer is run.
