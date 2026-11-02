"""Compute moving bistatic paths, antenna responses and endpoint truth."""
import json
import numpy as np
from scipy.interpolate import CubicSpline
from .parameters import FS,FC,C

def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def basis(direction, axis):
    d = unit(direction)
    a = np.broadcast_to(np.asarray(axis), d.shape).copy()
    e = a - np.sum(a * d, axis=-1, keepdims=True) * d
    bad = np.linalg.norm(e, axis=-1) < 1e-10
    if bad.any():
        a[bad] = np.eye(3)[np.argmin(abs(d[bad]), axis=-1)]
        e[bad] = a[bad] - np.sum(a[bad] * d[bad], axis=-1, keepdims=True) * d[bad]
    return unit(e)

def satellite(cfg, scene, time):
    t = np.asarray(time).reshape(-1)
    earth = cfg['satellite']['earth_radius_m']
    orbit = earth + cfg['satellite']['altitude_m']
    (el, az) = np.deg2rad([scene['satellite_elevation_deg'], scene['satellite_azimuth_deg']])
    slant = -earth * np.sin(el) + np.sqrt(orbit ** 2 - earth ** 2 * np.cos(el) ** 2)
    local = slant * np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    radial = np.array([earth + local[2], local[0], local[1]]) / orbit
    along = unit(np.array([1.0, 0, 0]) - radial[0] * radial)
    angle = np.deg2rad(scene.get('orbit_tangent_bearing_deg', 0))
    along = np.cos(angle) * along + np.sin(angle) * np.cross(radial, along)
    rate = np.sqrt(cfg['satellite']['mu_m3_s2'] / orbit ** 3)
    pos = orbit * (np.cos(rate * t)[:, None] * radial + np.sin(rate * t)[:, None] * along)
    vel = orbit * rate * (-np.sin(rate * t)[:, None] * radial + np.cos(rate * t)[:, None] * along)
    spin = cfg['satellite']['earth_rotation_rad_s']
    co = np.cos(spin * t)
    si = np.sin(spin * t)
    p = np.column_stack((co * pos[:, 0] + si * pos[:, 1], -si * pos[:, 0] + co * pos[:, 1], pos[:, 2]))
    v = np.column_stack((co * vel[:, 0] + si * vel[:, 1], -si * vel[:, 0] + co * vel[:, 1], vel[:, 2]))
    v -= np.column_stack((-spin * p[:, 1], spin * p[:, 0], np.zeros(len(t))))
    return (p[:, [1, 2, 0]] - np.array([0, 0, earth]), v[:, [1, 2, 0]])

def target(cfg, scene, time):
    t = np.asarray(time).reshape(-1)
    ground = np.asarray(scene['start_xy_m']) + t[:, None] * np.asarray(scene['velocity_xy_mps'])
    ground += 0.5 * t[:, None] ** 2 * np.asarray(scene['acceleration_xy_mps2'])
    r = np.linalg.norm(ground, axis=1)
    earth = cfg['satellite']['earth_radius_m']
    height = scene['height_m']
    horizontal = (earth + height) * np.sin(r / earth)
    return np.column_stack((ground * (horizontal / r)[:, None], (earth + height) * np.cos(r / earth) - earth))

def panel(cfg, index):
    r = cfg['receiver']
    az = np.deg2rad(r['azimuth_deg'][index])
    el = np.deg2rad(r['elevation_deg'][index])
    normal = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    right = np.array([np.cos(az), -np.sin(az), 0])
    up = np.cross(normal, right)
    n = r['elements_per_side']
    axis = (np.arange(n) - (n - 1) / 2) * r['spacing_m']
    (h, v) = np.meshgrid(axis, axis, indexing='ij')
    positions = h.ravel()[:, None] * right + v.ravel()[:, None] * up
    taper1 = np.sin(np.pi * (np.arange(n) + 0.5) / n) ** r['taper_exponent']
    taper = (taper1[:, None] * taper1[None]).ravel()
    taper /= np.linalg.norm(taper)
    return (np.array(r['panel_centres_m'][index]), normal, right, positions, taper)

def response(cfg, index, arrival, steer, electric):
    (_, normal, right, positions, taper) = panel(cfg, index)
    array = np.exp(2j * np.pi * FC / C * ((arrival - steer) @ positions.T)) @ taper
    element = np.pi * cfg['receiver']['aperture_efficiency'] * np.maximum(0, arrival @ normal)
    axis = cfg['receiver'].get('reference_polarization_axis', right) if index == 0 else right
    polarization = np.sum(electric * basis(-arrival, axis), axis=1)
    return np.sqrt(element) * polarization * array

def geometry(cfg, scene):
    times = np.arange(-20, 4021, dtype=float) * 0.001
    refresh = scene['steering_refresh_s']
    held = np.floor((times + 1e-10) / refresh) * refresh
    (refcentre, refnormal, _, _, _) = panel(cfg, 0)
    rem = times.copy()
    for _ in range(5):
        (sat, _) = satellite(cfg, scene, rem)
        rl = np.linalg.norm(sat - refcentre, axis=1)
        rem = times - rl / C
    refarrival = unit(sat - refcentre)
    (heldsat, _) = satellite(cfg, scene, held)
    refsteer = unit(heldsat - refcentre)
    refresponse = response(cfg, 0, refarrival, refsteer, basis(-refarrival, cfg['propagation']['tx_polarization_axis']))
    txpower = 10 ** (cfg['satellite']['eirp_density_dBW_MHz'] / 10) * 20
    noise = 1.380649e-23 * (cfg['receiver']['antenna_temperature_K'] + 290 * (10 ** (cfg['receiver']['noise_figure_dB'] / 10) - 1))
    rfield = np.sqrt(txpower * (C / FC) ** 2 / (4 * np.pi * rl) ** 2 / noise) * refresponse
    initial = target(cfg, scene, [0])[0]
    index = max(range(1, 5), key=lambda j: float(unit(initial - panel(cfg, j)[0]) @ panel(cfg, j)[1]))
    (centre, normal, _, _, _) = panel(cfg, index)
    bounce = times.copy()
    for _ in range(4):
        u = target(cfg, scene, bounce)
        ur = np.linalg.norm(u - centre, axis=1)
        bounce = times - ur / C
    emission = bounce.copy()
    for _ in range(5):
        (sat, _) = satellite(cfg, scene, emission)
        su = np.linalg.norm(sat - u, axis=1)
        emission = bounce - su / C
    arrival = unit(u - centre)
    heldu = target(cfg, scene, held)
    steer = unit(heldu - centre)
    incoming = unit(u - sat)
    outgoing = -arrival
    e = basis(incoming, cfg['propagation']['tx_polarization_axis'])
    e = np.sum(e * basis(incoming, cfg['propagation']['uav_copolar_axis']), axis=1)[:, None] * basis(outgoing, cfg['propagation']['uav_copolar_axis'])
    echo_response = response(cfg, index, arrival, steer, e)
    echo_power = txpower * (C / FC) ** 2 * 0.03 / ((4 * np.pi) ** 3 * su ** 2 * ur ** 2)
    efield = np.sqrt(echo_power / noise) * echo_response
    direct_emission = times.copy()
    for _ in range(5):
        (dsat, _) = satellite(cfg, scene, direct_emission)
        dl = np.linalg.norm(dsat - centre, axis=1)
        direct_emission = times - dl / C
    directarrival = unit(dsat - centre)
    direct_response = response(cfg, index, directarrival, steer, basis(-directarrival, cfg['propagation']['tx_polarization_axis']))
    dfield = np.sqrt(txpower * (C / FC) ** 2 / (4 * np.pi * dl) ** 2 / noise) * direct_response
    active = (times >= 0) & (times <= 4)
    earth = cfg['satellite']['earth_radius_m']
    minrange = float(np.linalg.norm(u[active, :2], axis=1).min())
    if minrange < 4999.999:
        raise AssertionError('Below 5 km exclusion radius')
    reference_elevation = np.rad2deg(np.arcsin(refarrival[:, 2]))
    if reference_elevation[active].min() < 35:
        raise AssertionError('Reference outside steering envelope')
    if np.any(steer[active] @ normal < np.cos(np.deg2rad(55))):
        raise AssertionError('UAV beam outside steering envelope')
    ref_emission = times - rl / C
    inverse = CubicSpline(ref_emission, times)
    delay = (times - inverse(times - (su + ur) / C)) * FS
    delayfit = CubicSpline(times, delay)
    fd = -FC / FS * delayfit(times, 1)
    rate = -FC / FS * delayfit(times, 2)
    labels = np.arange(1, 201) * 0.02 - 1 / FS
    summary = {'scene_id': scene['scene_id'], 'surveillance_panel': index, 'minimum_range_m': minrange, 'maximum_range_m': float(np.linalg.norm(u[active, :2], axis=1).max()), 'reference_elevation_deg': [float(reference_elevation[active].min()), float(reference_elevation[active].max())], 'doppler_Hz': [float(fd[active].min()), float(fd[active].max())], 'doppler_rate_Hz_s': [float(rate[active].min()), float(rate[active].max())], 'reference_snr_per_sample_dB': [float(10 * np.log10(np.min(abs(rfield[active]) ** 2) / FS)), float(10 * np.log10(np.max(abs(rfield[active]) ** 2) / FS))], 'beam_offset_deg': [float(np.rad2deg(np.arccos(np.clip(np.sum(arrival[active] * steer[active], axis=1), -1, 1))).min()), float(np.rad2deg(np.arccos(np.clip(np.sum(arrival[active] * steer[active], axis=1), -1, 1))).max())], 'beamformer_loss_convention': '1.5 dB common post-element signal/noise attenuation cancels in unit-noise voltage units', 'labels': [{'decision_s': float(t), 'delay_samples': float(delayfit(t)), 'frequency_Hz': float(-FC / FS * delayfit(t, 1)), 'rate_Hz_s': float(-FC / FS * delayfit(t, 2))} for t in labels]}
    import hashlib
    summary['geometry_fingerprint'] = hashlib.sha256(json.dumps({'cfg': cfg, 'scene': scene}, sort_keys=True).encode()).hexdigest()
    return ({'time_s': times, 'reference_length_m': rl, 'reference_field': rfield, 'echo_length_m': su + ur, 'echo_field': efield, 'direct_length_m': dl, 'direct_field': dfield, 'target_position_m': u}, summary)
