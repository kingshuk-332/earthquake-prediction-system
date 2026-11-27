"""
Validated configuration, feature registries, and repository-root discovery for the
earthquake-prediction-system remediated preprocessing and multi-modal fusion pipelines.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple


def resolve_repo_root(start: Path | None = None) -> Path:
    """
    Deterministically resolves the earthquake-prediction-system repository root
    without hardcoded developer workstation paths.
    """
    env_root = os.environ.get("EEW_REPO_ROOT")
    if env_root:
        cand = Path(env_root).expanduser().resolve()
        if (cand / "dataset_2.csv").is_file():
            return cand

    cur = (start or Path(__file__).resolve()).resolve()
    if cur.is_file():
        cur = cur.parent

    for parent in [cur, *cur.parents]:
        if (parent / "dataset_2.csv").is_file():
            return parent
        nested = parent / "earthquake-prediction-system"
        if (nested / "dataset_2.csv").is_file():
            return nested

    raise FileNotFoundError(
        "Could not locate repository root containing 'dataset_2.csv'. "
        "Set EEW_REPO_ROOT or run from within 'earthquake-prediction-system'."
    )


REQUIRED_CATALOG_COLUMNS: Tuple[str, ...] = (
    "year",
    "origin_time_utc",
    "Lat",
    "Long",
    "Depth_km",
    "Magnitude_Value",
    "Magnitude_Type",
    "source_id",
    "receiver_code",
    "receiver_latitude",
    "receiver_longitude",
    "receiver_elevation_m",
    "p_arrival_sample",
    "s_arrival_sample",
    "snr_db",
    "epicentral_dist_km",
)

BRANCH_A_FEATURE_NAMES: Tuple[str, ...] = (
    "past_mw_homogenized",
    "past_log1p_depth_km",
    "past_log1p_hypo_sph_km",
    "past_log1p_sp_sec",
    "past_snr_db",
    "past_log1p_interevent_hours",
)

BRANCH_B_FEATURE_NAMES: Tuple[str, ...] = (
    "station_ecef_x_norm",
    "station_ecef_y_norm",
    "station_ecef_z_norm",
    "station_elev_km",
    "station_log1p_count_30d",
    "station_log1p_count_90d",
    "station_max_mw_90d",
    "station_mean_snr_90d_scaled",
    "station_log1p_days_since_last_eq",
)

BRANCH_C_FEATURE_NAMES: Tuple[str, ...] = (
    "gr_b_value_180d",
    "log1p_benioff_30d",
    "log1p_benioff_180d",
    "log1p_etas_intensity",
    "log1p_rate_above_mc_7d",
    "log1p_rate_above_mc_30d",
    "sub_mc_fraction_30d",
    "origin_event_mw",
    "origin_event_log1p_depth_km",
    "origin_doy_sin",
    "origin_doy_cos",
    "origin_above_mc_flag",
)


@dataclass(frozen=True)
class GeophysicalConfig:
    earth_radius_km: float = 6371.0
    vp_km_s: float = 6.0
    vs_km_s: float = 3.5
    sampling_rate_hz: float = 100.0
    mc_completeness: float = 3.7
    lat_bounds: Tuple[float, float] = (-90.0, 90.0)
    lon_bounds: Tuple[float, float] = (-180.0, 180.0)
    depth_bounds_km: Tuple[float, float] = (0.0, 700.0)
    magnitude_bounds: Tuple[float, float] = (0.0, 10.0)
    snr_min_db: float = 0.0
    geodesic_kernel_length_km: float = 200.0


@dataclass(frozen=True)
class FusionModelConfig:
    lookback_events: int = 16
    kinematic_channels: int = 6
    gnn_node_dim: int = 9
    phys_feature_dim: int = 12
    hidden_dim: int = 64
    dropout: float = 0.15
    modality_dropout_prob: float = 0.30
    ensemble_seeds: Tuple[int, ...] = (42, 1337, 2026)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 24
    patience: int = 5
    tail_alpha: float = 0.75


@dataclass(frozen=True)
class ValidationConfig:
    modern_catalog_start_utc: str = "1990-01-01 00:00:00"
    forecast_horizon_days: float = 30.0
    embargo_days: float = 30.0
    n_walk_forward_folds: int = 5
    target_mw_threshold: float = 5.0
    evaluation_mw_thresholds: Tuple[float, ...] = (4.0, 4.5, 5.0, 5.5)
    min_positives_for_metric: int = 5
    holdout_stations: Tuple[str, ...] = ("KOH", "SHL")
    spatial_holdout_lon_min: float = 93.5
    spatial_buffer_deg: float = 0.5
    ece_bins: int = 10
    random_seed: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    dataset_rel_path: str = "dataset_2.csv"
    output_dir_rel_path: str = "seismic_fusion/outputs"
    geophysics: GeophysicalConfig = field(default_factory=GeophysicalConfig)
    model: FusionModelConfig = field(default_factory=FusionModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def validate(self) -> None:
        if self.validation.embargo_days < self.validation.forecast_horizon_days:
            raise ValueError(
                f"embargo_days ({self.validation.embargo_days}) must be >= "
                f"forecast_horizon_days ({self.validation.forecast_horizon_days}) to prevent temporal leakage."
            )
        if self.validation.target_mw_threshold not in self.validation.evaluation_mw_thresholds:
            raise ValueError("target_mw_threshold must be included in evaluation_mw_thresholds.")
        if self.geophysics.vs_km_s >= self.geophysics.vp_km_s or self.geophysics.vs_km_s <= 0:
            raise ValueError("Invalid seismic velocities: require 0 < vs_km_s < vp_km_s.")
        if self.geophysics.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be strictly positive.")
        if self.model.lookback_events < 2:
            raise ValueError("lookback_events must be >= 2.")

    @classmethod
    def from_json(cls, config_path: Path) -> "PipelineConfig":
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        geophys = GeophysicalConfig(
            earth_radius_km=float(raw["geophysics"]["earth_radius_km"]),
            vp_km_s=float(raw["geophysics"]["vp_km_s"]),
            vs_km_s=float(raw["geophysics"]["vs_km_s"]),
            sampling_rate_hz=float(raw["geophysics"]["sampling_rate_hz"]),
            mc_completeness=float(raw["geophysics"]["mc_completeness"]),
            lat_bounds=tuple(raw["geophysics"]["lat_bounds"]),
            lon_bounds=tuple(raw["geophysics"]["lon_bounds"]),
            depth_bounds_km=tuple(raw["geophysics"]["depth_bounds_km"]),
            magnitude_bounds=tuple(raw["geophysics"]["magnitude_bounds"]),
            snr_min_db=float(raw["geophysics"]["snr_min_db"]),
            geodesic_kernel_length_km=float(raw["geophysics"]["geodesic_kernel_length_km"]),
        )
        mod = FusionModelConfig(
            lookback_events=int(raw["model"]["lookback_events"]),
            kinematic_channels=int(raw["model"]["kinematic_channels"]),
            gnn_node_dim=int(raw["model"]["gnn_node_dim"]),
            phys_feature_dim=int(raw["model"]["phys_feature_dim"]),
            hidden_dim=int(raw["model"]["hidden_dim"]),
            dropout=float(raw["model"]["dropout"]),
            modality_dropout_prob=float(raw["model"].get("modality_dropout_prob", 0.30)),
            ensemble_seeds=tuple(int(s) for s in raw["model"]["ensemble_seeds"]),
            learning_rate=float(raw["model"]["learning_rate"]),
            weight_decay=float(raw["model"]["weight_decay"]),
            batch_size=int(raw["model"]["batch_size"]),
            max_epochs=int(raw["model"]["max_epochs"]),
            patience=int(raw["model"]["patience"]),
            tail_alpha=float(raw["model"]["tail_alpha"]),
        )
        val_raw = raw["validation"]
        eval_ths = tuple(
            float(x)
            for x in val_raw.get("evaluation_mw_thresholds", val_raw.get("secondary_mw_thresholds", [4.0, 4.5, 5.0, 5.5]))
        )
        val = ValidationConfig(
            modern_catalog_start_utc=str(val_raw["modern_catalog_start_utc"]),
            forecast_horizon_days=float(val_raw["forecast_horizon_days"]),
            embargo_days=float(val_raw["embargo_days"]),
            n_walk_forward_folds=int(val_raw["n_walk_forward_folds"]),
            target_mw_threshold=float(val_raw["target_mw_threshold"]),
            evaluation_mw_thresholds=eval_ths,
            min_positives_for_metric=int(val_raw.get("min_positives_for_metric", 5)),
            holdout_stations=tuple(str(s) for s in val_raw["holdout_stations"]),
            spatial_holdout_lon_min=float(val_raw.get("spatial_holdout_lon_min", 93.5)),
            spatial_buffer_deg=float(val_raw.get("spatial_buffer_deg", 0.5)),
            ece_bins=int(val_raw["ece_bins"]),
            random_seed=int(val_raw["random_seed"]),
        )
        cfg = cls(
            dataset_rel_path=str(raw.get("dataset_rel_path", "dataset_2.csv")),
            output_dir_rel_path=str(raw.get("output_dir_rel_path", "seismic_fusion/outputs")),
            geophysics=geophys,
            model=mod,
            validation=val,
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
