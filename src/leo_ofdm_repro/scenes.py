"""Construct deterministic satellite and UAV scenes."""
import hashlib
import numpy as np
COUNTS={'train':(16,2,40),'validation':(8,1,100),'test':(8,1,100)}

def seed(*parts):
    value = '|'.join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], 'little') & 2147483647

def make_scene(split, index, null=False):
    rng = np.random.default_rng(seed('sliding-v1', split, 'null' if null else 'target', index))
    group = index % 4
    elevations = (35.5, 48.0, 60.5, 73.0)
    elevation = elevations[group] + rng.uniform(0.5, 11.0)
    altitude_counts = {'train': [25] * 6 + [50] * 5 + [100] * 5, 'validation': [25] * 3 + [50] * 3 + [100] * 2, 'test': [25] * 2 + [50] * 3 + [100] * 3}
    levels = np.asarray(altitude_counts[split])
    np.random.default_rng(seed(split, 'altitude-order')).shuffle(levels)
    height = float(rng.choice([25, 50, 100])) if null else float(levels[index])
    for _ in range(10000):
        az = (group * 90 + rng.uniform(5, 85)) % 360
        range_m = rng.uniform(5200, 7650)
        block = index // 4 % 4
        family = group ^ [0, 2, 3, 1][block] ^ list(COUNTS).index(split)
        heading = az + (180, 0, 90, 45)[family] + rng.uniform(-25, 25)
        speed = rng.uniform(45, 79.5)
        acceleration = rng.uniform(-1.5, 1.5)
        a = np.deg2rad(az)
        h = np.deg2rad(heading)
        start = range_m * np.array([np.sin(a), np.cos(a)])
        velocity = speed * np.array([np.sin(h), np.cos(h)])
        accel = acceleration * np.array([np.cos(h), -np.sin(h)])
        t = np.linspace(0, 4, 401)
        positions = start[None] + t[:, None] * velocity + 0.5 * t[:, None] ** 2 * accel
        ranges = np.linalg.norm(positions, axis=1)
        speeds = np.linalg.norm(velocity[None] + t[:, None] * accel, axis=1)
        if ranges.min() >= 5000 and ranges.max() <= 8000 and (speeds.min() >= 45) and (speeds.max() <= 80):
            break
    else:
        raise RuntimeError('Could not sample valid flight')
    return {'scene_id': f"{split}_{('null' if null else 'path')}_{index:03d}", 'split': split, 'index': index, 'null_scene': null, 'waveform_id': group ^ block, 'height_m': height, 'start_xy_m': start.tolist(), 'velocity_xy_mps': velocity.tolist(), 'acceleration_xy_mps2': accel.tolist(), 'satellite_elevation_deg': float(elevation), 'satellite_azimuth_deg': float((index * 137.507764 + rng.uniform(-20, 20)) % 360), 'orbit_tangent_bearing_deg': float(rng.uniform(0, 360)), 'scattering_phase_rad': float(rng.uniform(-np.pi, np.pi)), 'steering_refresh_s': 0.1}
