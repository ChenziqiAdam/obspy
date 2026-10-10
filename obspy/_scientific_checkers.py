"""Opt-in scientific runtime checks used by the SciBench pilot.

The module is deliberately inert unless ``SCIBENCH_TRIGGER_LOG`` names a log
file.  Checks append stable IDs and never raise into ObsPy.
"""

from __future__ import annotations

import json
import os
import warnings
from contextlib import contextmanager

import numpy as np


_ACTIVE = False
_EPS = np.finfo(np.float64).eps


def enabled():
    return bool(os.environ.get("SCIBENCH_TRIGGER_LOG"))


def trigger(checker_id):
    path = os.environ.get("SCIBENCH_TRIGGER_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"checker_id": checker_id}) + "\n")
    except Exception:
        pass


def trigger_if(condition, checker_id):
    if bool(condition):
        trigger(checker_id)


@contextmanager
def _checking():
    global _ACTIVE
    if _ACTIVE:
        yield False
        return
    _ACTIVE = True
    try:
        # A checker re-calls public APIs on transformed inputs; warnings raised
        # by those calls must never reach the user's code.
        # (inserted directly: catch_warnings would reset every module's
        # once-per-location warning registry)
        flt = ("ignore", None, Warning, None, 0)
        warnings.filters.insert(0, flt)
        try:
            with np.errstate(all="ignore"):
                yield True
        finally:
            try:
                warnings.filters.remove(flt)
            except ValueError:
                pass
    finally:
        _ACTIVE = False


def _finite(*values):
    return all(np.all(np.isfinite(np.asarray(value))) for value in values)


def _scale(*values):
    result = 1.0
    for value in values:
        array = np.asarray(value)
        if array.size:
            result = max(result, float(np.max(np.abs(array))))
    return result


def _dtype_epsilon(*values):
    """Largest machine epsilon among represented floating-point dtypes."""
    epsilon = _EPS
    for value in values:
        dtype = np.asarray(value).dtype
        if dtype.kind in "fc":
            epsilon = max(epsilon, float(np.finfo(dtype).eps))
    return epsilon


def _different(a, b, factor=256.0, condition=1.0, epsilon=None):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or not _finite(a, b):
        return True
    if epsilon is None:
        epsilon = _dtype_epsilon(a, b)
    tol = factor * epsilon * max(1.0, float(condition)) * _scale(a, b)
    return bool(np.any(np.abs(a - b) > tol))


def _different_at_scale(a, b, reference_scale, factor=256.0,
                        condition=1.0, epsilon=None):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or not _finite(a, b, reference_scale):
        return True
    if epsilon is None:
        epsilon = _dtype_epsilon(a, b)
    tol = (factor * epsilon * max(1.0, float(condition)) *
           max(1.0, float(reference_scale)))
    return bool(np.any(np.abs(a - b) > tol))


def _unmasked(*values):
    """Plain arrays restricted to samples unmasked in every value."""
    masks = [np.ma.getmaskarray(v) for v in values]
    if not any(m.any() for m in masks):
        return tuple(np.ma.getdata(v) for v in values)
    keep = ~np.logical_or.reduce(masks)
    return tuple(np.ma.getdata(v)[keep] for v in values)


def check_rotate_ne_rt(n, e, ba, r, t):
    try:
        with _checking() as run:
            if not run or not _finite(n, e, r, t, ba):
                return
            n, e, r, t = _unmasked(n, e, r, t)
            scale = _scale(n, e, r, t)
            if scale > np.sqrt(np.finfo(float).max) / 8:
                return
            for v in (n, e):
                v = np.asarray(v)
                # unary minus wraps for unsigned ints and for iinfo.min
                if v.dtype.kind == "u":
                    return
                if v.dtype.kind == "i" and v.size and \
                        np.any(v == np.iinfo(v.dtype).min):
                    return
            from obspy.signal.rotate import rotate_rt_ne
            n2, e2 = rotate_rt_ne(r, t, ba)
            in_eps = _dtype_epsilon(n, e)  # channels may differ in precision
            trigger_if(_different_at_scale(n, n2, scale, 64, epsilon=in_eps) or
                       _different_at_scale(e, e2, scale, 64, epsilon=in_eps),
                       "OB-ROT-001")
            lhs = np.asarray(n, float) ** 2 + np.asarray(e, float) ** 2
            rhs = np.asarray(r, float) ** 2 + np.asarray(t, float) ** 2
            trigger_if(_different(lhs, rhs, 128,
                                  epsilon=_dtype_epsilon(n, e, r, t)),
                       "OB-ROT-002")
    except Exception:
        pass


def check_rotate_zne_lqt(z, n, e, ba, inc, l, q, t):
    try:
        with _checking() as run:
            if not run:
                return
            z, n, e, l, q, t = _unmasked(z, n, e, l, q, t)
            if not _finite(z, n, e, l, q, t, ba, inc):
                return
            scale = _scale(z, n, e, l, q, t)
            if scale > np.sqrt(np.finfo(float).max) / 8:
                return
            from obspy.signal.rotate import rotate_lqt_zne
            z2, n2, e2 = rotate_lqt_zne(l, q, t, ba, inc)
            in_eps = _dtype_epsilon(z, n, e)
            bad = (_different_at_scale(z, z2, scale, 96, epsilon=in_eps) or
                   _different_at_scale(n, n2, scale, 96, epsilon=in_eps) or
                   _different_at_scale(e, e2, scale, 96, epsilon=in_eps))
            trigger_if(bad, "OB-ROT-003")
            lhs = (np.asarray(z, float) ** 2 + np.asarray(n, float) ** 2 +
                   np.asarray(e, float) ** 2)
            rhs = (np.asarray(l, float) ** 2 + np.asarray(q, float) ** 2 +
                   np.asarray(t, float) ** 2)
            trigger_if(_different(lhs, rhs, 192,
                                  epsilon=_dtype_epsilon(z, n, e, l, q, t)),
                       "OB-ROT-004")
    except Exception:
        pass


def check_rotate2zne(data, orientations, inverse, result):
    try:
        with _checking() as run:
            if not run or inverse:
                return
            n_data = len(data)
            both = _unmasked(*data, *result)
            data, result = both[:n_data], both[n_data:]
            if not _finite(*data, *result):
                return
            if any(np.asarray(v).dtype.kind == "u" for v in data):
                # unary minus / differences wrap for unsigned counts
                return
            from obspy.signal.rotate import (_dip_azimuth2zne_base_vector,
                                             rotate2zne)
            matrix = np.array([
                _dip_azimuth2zne_base_vector(dip, az)
                for az, dip in orientations
            ])
            condition = float(np.linalg.cond(matrix))
            if not np.isfinite(condition) or condition > 1e8:
                return
            args = []
            for component, (azimuth, dip) in zip(result, orientations):
                args.extend((component, azimuth, dip))
            restored = rotate2zne(*args, inverse=True)
            signal_scale = _scale(*data, *result, *restored)
            bad = any(_different_at_scale(a, b, signal_scale, 256, condition)
                      for a, b in zip(data, restored))
            trigger_if(bad, "OB-ROT-005")

            order = (1, 2, 0)
            args = []
            for index in order:
                args.extend((data[index], orientations[index][0],
                             orientations[index][1]))
            permuted = rotate2zne(*args)
            bad = any(_different_at_scale(a, b, signal_scale, 256, condition)
                      for a, b in zip(result, permuted))
            trigger_if(bad, "OB-ROT-006")
    except Exception:
        pass


_ARRAY_KEYS = ("ts_d", "ts_dh", "ts_s", "ts_sh", "ts_wmag", "ts_w1",
               "ts_w2", "ts_w3", "ts_tilt", "ts_e")


def _array_result_differs(first, second, condition, extra=1.0, epsilon=None):
    return any(_different(first[key], second[key], 2048 * extra, condition)
               if epsilon is None else
               _different(first[key], second[key], 2048 * extra, condition,
                          epsilon=epsilon)
               for key in _ARRAY_KEYS)


def check_array_rotation_strain(subarray, ts1, ts2, ts3, vp, vs,
                                array_coords, sigmau, result):
    try:
        with _checking() as run:
            if not run or not _finite(ts1, ts2, ts3, array_coords, vp, vs,
                                      sigmau):
                return
            if not (vp > vs > 0):
                return
            # station differences of unsigned counts wrap in the library
            if any(np.asarray(t).dtype.kind == "u" for t in (ts1, ts2, ts3)):
                return
            condition = float(np.linalg.cond(result["A"].T @ result["A"]))
            if not np.isfinite(condition) or condition > 1e7:
                return
            from obspy.signal.array_analysis import array_rotation_strain
            na = np.asarray(array_coords).shape[0]
            order = np.arange(na)[::-1]
            inverse = np.empty(na, dtype=int)
            inverse[order] = np.arange(na)
            sigma = np.asarray(sigmau)
            if sigma.ndim == 0:
                sigma_p = sigmau
            else:
                sigma_p = sigma[order]
            permuted = array_rotation_strain(
                inverse[np.asarray(subarray, dtype=int)],
                np.asarray(ts1)[:, order], np.asarray(ts2)[:, order],
                np.asarray(ts3)[:, order], vp, vs,
                np.asarray(array_coords)[order], sigma_p)
            trigger_if(_array_result_differs(result, permuted, condition),
                       "OB-ARR-001")

            coords = np.asarray(array_coords)
            used = np.asarray(subarray, dtype=int)
            # only the stations in the sub-array enter the solution
            used_coords = coords[used]
            span = float(np.ptp(used_coords, axis=0).max())
            if span == 0:
                return
            max_coord = _scale(used_coords)
            if max_coord / span < 1e6:
                shift = span * np.array([0.5, -0.25, 0.125],
                                        dtype=coords.dtype)
                translated = array_rotation_strain(
                    subarray, ts1, ts2, ts3, vp, vs, coords + shift, sigmau)
                # coords + shift re-rounds every coordinate by
                # eps*max_coord, a relative geometry change of
                # eps*max_coord/span
                trigger_if(_array_result_differs(
                               result, translated, condition,
                               4.0 * max(1.0, max_coord / span),
                               _dtype_epsilon(coords)),
                           "OB-ARR-002")

            used_data = [np.asarray(t)[:, used] for t in (ts1, ts2, ts3)]
            data_scale = max(_scale(*used_data) if used.size else 1.0, 0.0)
            # adding common motion re-rounds every sample by eps*data_scale,
            # a relative change eps*data_scale/signal of the station
            # differences that carry the gradient
            signal = max(float(np.max(np.ptp(t, axis=1))) for t in used_data)
            if signal > 0 and data_scale / signal < 1e8 and \
                    data_scale < np.finfo(float).max / 8:
                requantisation = max(1.0, data_scale / signal)
                nt = np.asarray(ts1).shape[0]
                common = np.linspace(-0.25, 0.25, nt) * data_scale
                common_result = array_rotation_strain(
                    subarray, np.asarray(ts1) + common[:, None],
                    np.asarray(ts2) - 0.5 * common[:, None],
                    np.asarray(ts3) + 0.25 * common[:, None], vp, vs,
                    array_coords, sigmau)
                trigger_if(_array_result_differs(
                               result, common_result, condition,
                               8.0 * requantisation,
                               _dtype_epsilon(ts1, ts2, ts3)),
                           "OB-ARR-003")
    except Exception:
        pass


def _moment_components(matrix):
    return (matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[0, 1],
            matrix[0, 2], matrix[1, 2])


def check_farfield(mt, points, wave_type, displacement):
    try:
        with _checking() as run:
            if not run or not _finite(mt, points, displacement):
                return
            points = np.asarray(points)
            if (points.dtype.kind in "iu" and points.ndim == 2 and
                    points.shape[0] == 3 and points.size and
                    np.max(np.sum(points.astype(float) ** 2, axis=0)) >
                    np.iinfo(points.dtype).max):
                # farfield squares integer vectors in their own dtype
                return
            epsilon = _dtype_epsilon(mt, points, displacement)
            if points.dtype.kind in "iu":
                # farfield's np.sqrt promotes int8->float16, int16->float32
                promoted = np.sqrt(np.zeros(1, dtype=points.dtype)).dtype
                epsilon = max(epsilon, float(np.finfo(promoted).eps))
            if points.shape[0] == 2:
                cart = np.vstack((np.sin(points[0]) * np.cos(points[1]),
                                  np.sin(points[0]) * np.sin(points[1]),
                                  np.cos(points[0])))
            elif points.shape[0] == 3:
                cart = points
            else:
                return
            lengths = np.linalg.norm(cart, axis=0)
            if np.any(lengths == 0) or not np.all(np.isfinite(lengths)):
                return
            gamma = cart / lengths
            disp = np.asarray(displacement, dtype=float)
            tensor_scale = _scale(mt)
            tol = 512 * epsilon * max(1.0, tensor_scale, _scale(disp))
            radial = np.sum(disp * gamma, axis=0)
            if str(wave_type).upper() == "P":
                transverse = disp - gamma * radial
                trigger_if(np.any(np.abs(transverse) > tol), "OB-SRC-001")
            else:
                trigger_if(np.any(np.abs(radial) > tol), "OB-SRC-002")

            from obspy.core.event.source import _fullmt, farfield
            if points.shape[0] == 2:
                other = farfield(mt, cart, wave_type)
                # np.sin of int8/int16 angles is evaluated in float16/32
                trig_eps = float(np.finfo(np.result_type(points.dtype,
                                                         np.float16)).eps)
                trigger_if(_different(disp, other, 512,
                                      epsilon=max(epsilon, trig_eps)),
                           "OB-SRC-003")
            elif not (points.dtype.kind == "u" or
                      (points.dtype.kind == "i" and
                       np.any(points == np.iinfo(points.dtype).min))):
                # -points must be representable in the integer dtype
                other = farfield(mt, -points, wave_type)
                trigger_if(_different(other, -disp, 512, epsilon=epsilon),
                           "OB-SRC-004")

            cart_dtype = cart.dtype if cart.dtype.kind == "f" else float
            rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]],
                                dtype=cart_dtype)
            matrix = _fullmt(mt)
            rotated_mt = _moment_components(rotation @ matrix @ rotation.T)
            rotated = farfield(rotated_mt, rotation @ cart, wave_type)
            trigger_if(_different(rotated, rotation @ disp, 1024,
                                  epsilon=epsilon),
                       "OB-SRC-005")
    except Exception:
        pass


def _angle_difference_mod_180(a, b):
    return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)


def check_flinn(stream, noise_thres, result):
    try:
        with _checking() as run:
            if not run or not _finite(*stream, noise_thres, result):
                return
            arrays = [np.asarray(value) for value in stream]
            if min(value.size for value in arrays) < 3:
                return
            from obspy.signal.polarization import flinn
            if any(value.dtype.kind not in "f" for value in arrays):
                # integer counts: evaluate every law in float64 so neither
                # integer wrap-around nor unrepresentable negation enters
                arrays = [value.astype(np.float64) for value in arrays]
                result = flinn(arrays, noise_thres)
            # the library classifies samples by energy in the input dtype;
            # skip when that classification is not determined at input
            # precision (energy within rounding of the threshold, incl.
            # subnormal float16/32 squares)
            energy64 = sum(value.astype(np.float64) ** 2 for value in arrays)
            in_eps = _dtype_epsilon(*arrays)
            in_tiny = max(float(np.finfo(value.dtype).tiny)
                          for value in arrays)
            margin = 16 * in_eps * (np.maximum(energy64, abs(noise_thres)) +
                                    in_tiny)
            if np.any(np.abs(energy64 - noise_thres) <= margin):
                return
            mask = sum(value ** 2 for value in arrays) > noise_thres
            if np.count_nonzero(mask) < 3:
                return
            covariance = np.cov(np.vstack([arrays[2][mask], arrays[1][mask],
                                            arrays[0][mask]]))
            eigenvalues = np.linalg.eigvalsh(covariance)
            eig_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
            eig_gap = min(eigenvalues[2] - eigenvalues[1],
                          eigenvalues[1] - eigenvalues[0])
            if eig_gap <= 1e-8 * eig_scale:
                return
            scaled = flinn([2 * value for value in arrays], 4 * noise_thres)
            bad = (_angle_difference_mod_180(result[0], scaled[0]) > 1e-8 or
                   np.any(np.abs(np.asarray(result[1:]) -
                                 np.asarray(scaled[1:])) > 1e-10))
            trigger_if(bad, "OB-POL-001")

            rotated = flinn([arrays[0], arrays[2], -arrays[1]], noise_thres)
            expected_azimuth = (float(result[0]) - 90.0) % 180.0
            # azimuth is undefined for a (near-)vertical principal axis
            azimuth_defined = min(float(result[1]), float(rotated[1])) > 1e-3
            bad = ((azimuth_defined and
                    _angle_difference_mod_180(rotated[0], expected_azimuth) >
                    1e-7) or np.any(np.abs(np.asarray(result[1:]) -
                                        np.asarray(rotated[1:])) > 1e-9))
            trigger_if(bad, "OB-POL-002")

            negated = flinn([-value for value in arrays], noise_thres)
            bad = (_angle_difference_mod_180(result[0], negated[0]) > 1e-8 or
                   np.any(np.abs(np.asarray(result[1:]) -
                                 np.asarray(negated[1:])) > 1e-10))
            trigger_if(bad, "OB-POL-003")
    except Exception:
        pass


def check_eigval(datax, datay, dataz, fk, normf, result):
    try:
        with _checking() as run:
            if not run or not _finite(datax, datay, dataz, fk, normf,
                                      *result) or normf <= 0:
                return
            from obspy.signal.polarization import eigval
            # float64 first: integer negation / doubling can wrap
            dx, dy, dz = (np.asarray(v, dtype=np.float64)
                          for v in (datax, datay, dataz))
            rotated = eigval(dy, -dx, dz, fk, normf)
            # eigenvalue outputs carry absolute error ~eps*largest eigenvalue
            lam = max(1.0, _scale(result[0], result[1], result[2]))
            eig_idx = (0, 1, 2, 5)
            # outputs 5-7 are FIR time derivatives with gain sum|fk|
            gain = max(1.0, float(np.sum(np.abs(fk))))
            trigger_if(any(
                _different_at_scale(a, b, lam * (gain if i == 5 else 1.0),
                                    4096) if i in eig_idx
                else _different(a, b, 4096 * (gain if i > 5 else 1.0))
                for i, (a, b) in enumerate(zip(result, rotated))),
                "OB-POL-004")

            scaled = eigval(2 * dx, 2 * dy, 2 * dz, fk, normf)
            bad = any(_different_at_scale(
                          4 * np.asarray(result[index]), scaled[index],
                          4 * lam * (gain if index == 5 else 1.0), 4096)
                      for index in (0, 1, 2, 5))
            bad = bad or any(_different(result[index], scaled[index],
                                        4096 * (gain if index > 5 else 1.0))
                             for index in (3, 4, 6, 7))
            trigger_if(bad, "OB-POL-005")
    except Exception:
        pass


def _arrivals_differ(reference, candidate, tolerance):
    if len(reference) != len(candidate):
        return True
    for left, right in zip(reference, candidate):
        if left.name != right.name:
            return True
        scale = max(1.0, abs(left.time), abs(right.time),
                    abs(left.ray_param), abs(right.ray_param))
        tol = max(10 * float(tolerance), 4096 * _EPS * scale)
        if abs(left.time - right.time) > tol:
            return True
        if abs(left.ray_param - right.ray_param) > tol:
            return True
    return False


def check_taup_travel_times(model, source_depth, distance, phase_list,
                            receiver_depth, ray_param_tol, arrivals):
    try:
        with _checking() as run:
            if not run or distance is None or not arrivals:
                return
            rays = model.get_ray_paths(source_depth, distance, phase_list,
                                       receiver_depth, ray_param_tol)
            trigger_if(_arrivals_differ(arrivals, rays, ray_param_tol),
                       "OB-TAUP-001")
            pierce = model.get_pierce_points(source_depth, distance, phase_list,
                                             receiver_depth, [], ray_param_tol)
            trigger_if(_arrivals_differ(arrivals, pierce, ray_param_tol),
                       "OB-TAUP-002")
    except Exception:
        pass


def _subnormal_energy(d64):
    """True when squared samples approach the float64 subnormal range."""
    if not d64.size:
        return False
    peak = float(np.max(np.abs(d64)))
    return 0 < peak * peak < np.finfo(float).tiny / _EPS


def check_classic_sta_lta(data, nsta, nlta, result):
    try:
        with _checking() as run:
            if not run or not _finite(data, result) or not (nlta > nsta > 0):
                return
            from obspy.signal.trigger import classic_sta_lta_py
            d64 = np.asarray(data, dtype=float)
            if _subnormal_energy(d64):
                return
            other = classic_sta_lta_py(d64, nsta, nlta)
            result = np.asarray(result)
            if result.shape != other.shape or not _finite(other):
                trigger("OB-TRIG-001")
                return
            # running-sum cancellation: the error at sample t is
            # ~ eps * cumulative energy(t) / energy of the STA window(t)
            energy = d64 ** 2
            cumulative = np.cumsum(energy)
            window = np.convolve(energy, np.ones(int(nsta)))[:len(energy)]
            with np.errstate(all="ignore"):
                cond = np.where(window > 0, cumulative / window, np.inf)
            tol = (64 * len(d64) * _dtype_epsilon(result, other) *
                   np.maximum(1.0, cond) *
                   np.maximum(1.0, np.maximum(np.abs(result), np.abs(other))))
            trigger_if(np.any(np.abs(result - other) > tol), "OB-TRIG-001")
    except Exception:
        pass


def check_recursive_sta_lta(data, nsta, nlta, result):
    try:
        with _checking() as run:
            if not run or not _finite(data, result) or not (nlta > nsta > 0):
                return
            from obspy.signal.trigger import recursive_sta_lta_py
            d64 = np.asarray(data, dtype=float)
            # STA/LTA is scale invariant; normalise so the reference's
            # tiny-initialised LTA is not biased for physically tiny units
            peak = float(np.max(np.abs(d64))) if d64.size else 0.0
            if _subnormal_energy(d64):
                return
            other = recursive_sta_lta_py(d64 / peak if peak > 0 else d64,
                                         nsta, nlta)
            trigger_if(_different(result, other, 64 * max(nlta, len(data))),
                       "OB-TRIG-002")
    except Exception:
        pass


def check_paz_response(poles, zeros, scale_fac, t_samp, nfft, response,
                       frequencies):
    try:
        with _checking() as run:
            if not run or not _finite(poles, zeros, scale_fac, t_samp,
                                      response, frequencies):
                return
            if len(response) == 0:
                return
            from obspy.signal.invsim import paz_2_amplitude_value_of_freq_resp
            indices = np.unique(np.linspace(0, len(response) - 1,
                                            min(33, len(response)),
                                            dtype=int))
            bad = False
            for index in indices:
                s = 2j * np.pi * frequencies[index]
                conditions = []
                for roots in (zeros, poles):
                    if len(roots) == 0:
                        conditions.append(1.0)
                        continue
                    coefficients = np.poly(roots)
                    value = np.polyval(coefficients, s)
                    upper = np.polyval(np.abs(coefficients), abs(s))
                    condition = abs(upper) / max(abs(value),
                                                 np.finfo(float).tiny)
                    conditions.append(float(condition))
                condition = max(1.0, *conditions)
                if not np.isfinite(condition) or condition > 1e8:
                    continue
                # evaluate the reference in complex128; allow the precision
                # of the supplied pole/zero dtype
                expected = paz_2_amplitude_value_of_freq_resp(
                    {"poles": list(np.asarray(poles, dtype=complex)),
                     "zeros": list(np.asarray(zeros, dtype=complex)),
                     "gain": abs(scale_fac)}, frequencies[index])
                observed = abs(response[index])
                degree = max(1, len(poles), len(zeros))
                tol = (8 * degree * condition *
                       _dtype_epsilon(poles, zeros, response) *
                       max(1.0, expected, observed))
                if abs(expected - observed) > tol:
                    bad = True
                    break
            trigger_if(bad, "OB-RESP-001")
    except Exception:
        pass


def _circular_difference(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def check_gps2dist_azimuth(lat1, lon1, lat2, lon2, a, f, result):
    try:
        with _checking() as run:
            if not run or not _finite(lat1, lon1, lat2, lon2, a, f, result):
                return
            if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90 and
                    a > 0 and 0 <= f < 1 and result[0] > 0):
                return
            from obspy.geodetics.base import (gps2dist_azimuth,
                                               locations2degrees)
            reverse = gps2dist_azimuth(lat2, lon2, lat1, lon1, a=a, f=f)
            distance_scale = max(1.0, float(a), abs(result[0]),
                                 abs(reverse[0]))
            distance_tol = 4096 * _EPS * distance_scale
            trigger_if(abs(result[0] - reverse[0]) > distance_tol,
                       "OB-GEO-001")
            # Endpoint bearings become ill-conditioned as angular separation
            # approaches zero.  A longitude-frame shift first incurs input
            # rounding, which is amplified by the inverse angular separation.
            angular_separation = min(np.pi, max(0.0, result[0] / a))
            singular_separation = max(
                min(angular_separation, np.pi - angular_separation), _EPS)
            coordinate_scale = np.deg2rad(
                max(360.0, abs(lon1), abs(lon2),
                    abs(lon1 + 360.0), abs(lon2 + 360.0)))
            bearing_condition = max(1.0,
                                    coordinate_scale / singular_separation)
            angle_tol = max(4096.0, 64.0 * bearing_condition) * \
                _EPS * 360.0
            if bearing_condition <= 1e8:
                bad_bearings = (
                    _circular_difference(result[1], reverse[2]) > angle_tol or
                    _circular_difference(result[2], reverse[1]) > angle_tol)
                trigger_if(bad_bearings, "OB-GEO-002")

            if (bearing_condition <= 1e8 and
                    max(abs(lon1), abs(lon2)) <= 1e6):
                # shift the stored values in float64 (no dtype re-rounding)
                shifted = gps2dist_azimuth(float(lat1), float(lon1) + 360.0,
                                           float(lat2), float(lon2) + 360.0,
                                           a=a, f=f)
                bad_shift = (abs(result[0] - shifted[0]) > distance_tol or
                             _circular_difference(result[1], shifted[1]) >
                             angle_tol or
                             _circular_difference(result[2], shifted[2]) >
                             angle_tol)
                trigger_if(bad_shift, "OB-GEO-003")

            spherical = result if f == 0 else gps2dist_azimuth(
                lat1, lon1, lat2, lon2, a=a, f=0)
            angular = locations2degrees(float(lat1), float(lon1),
                                        float(lat2), float(lon2))
            expected = np.deg2rad(angular) * a
            sphere_tol = 8192 * _EPS * max(1.0, float(a),
                                           abs(float(expected)))
            trigger_if(abs(spherical[0] - expected) > sphere_tol,
                       "OB-GEO-006")
    except Exception:
        pass


def check_locations2degrees(lat1, lon1, lat2, lon2, result):
    try:
        with _checking() as run:
            if not run or not _finite(lat1, lon1, lat2, lon2, result):
                return
            lat1 = np.asarray(lat1)
            lat2 = np.asarray(lat2)
            lon1 = np.asarray(lon1)
            lon2 = np.asarray(lon2)
            if (np.any(np.abs(lat1) > np.pi / 2) or
                    np.any(np.abs(lat2) > np.pi / 2) or
                    _scale(lon1, lon2) > np.deg2rad(1e5)):
                return
            from obspy.geodetics.base import locations2degrees
            # Inputs at this observation point are already in radians.
            lat1d, lat2d = np.degrees(lat1), np.degrees(lat2)
            lon1d, lon2d = np.degrees(lon1), np.degrees(lon2)
            reverse = locations2degrees(lat2d, lon2d, lat1d, lon1d)
            epsilon = _dtype_epsilon(lat1, lon1, lat2, lon2, result)
            # a longitude of magnitude L is only known to eps*L
            lon_deg = float(np.degrees(_scale(lon1, lon2)))
            tol = 4096 * epsilon * 180.0 * max(1.0, lon_deg / 180.0)
            trigger_if(np.any(np.abs(np.asarray(result) - reverse) > tol),
                       "OB-GEO-004")
            shifted = locations2degrees(lat1d, lon1d + 90.0,
                                         lat2d, lon2d + 90.0)
            trigger_if(np.any(np.abs(np.asarray(result) - shifted) > tol),
                       "OB-GEO-005")
    except Exception:
        pass


def check_mean_longitude(longitudes, result):
    try:
        with _checking() as run:
            original_values = np.asarray(longitudes)
            values = np.asarray(longitudes, dtype=float)
            if (not run or values.size == 0 or not _finite(values, result) or
                    np.any(np.abs(values) > 180)):
                return
            resultant = abs(np.mean(np.exp(1j * np.radians(values))))
            if resultant <= 1e-8:
                return
            shifted_values = (values + 90.0 + 180.0) % 360.0 - 180.0
            from obspy.geodetics.base import mean_longitude
            shifted = mean_longitude(shifted_values)
            expected = (float(result) + 90.0 + 180.0) % 360.0 - 180.0
            epsilon = _dtype_epsilon(original_values, result)
            tol = 4096 * epsilon * max(1, values.size) / resultant * 360.0
            trigger_if(_circular_difference(shifted, expected) > tol,
                       "OB-GEO-007")
    except Exception:
        pass


def check_correlate(a, b, shift, normalize, method, result):
    try:
        with _checking() as run:
            a = np.asarray(a)
            b = np.asarray(b)
            if (not run or a.ndim != 1 or b.ndim != 1 or
                    len(a) != len(b) or len(a) == 0 or
                    not np.issubdtype(a.dtype, np.floating) or
                    not np.issubdtype(b.dtype, np.floating) or
                    not _finite(a, b, result)):
                return
            scale_a = float(np.max(np.abs(a)))
            scale_b = float(np.max(np.abs(b)))
            if scale_a == 0 or scale_b == 0:
                return
            scaled_norm = np.sqrt(np.sum((a / scale_a) ** 2) *
                                  np.sum((b / scale_b) ** 2))
            if (not np.isfinite(scaled_norm) or
                    scale_a > np.sqrt(np.finfo(float).max / len(a)) or
                    scale_b > np.sqrt(np.finfo(float).max / len(b)) or
                    scale_a * scale_b > np.finfo(float).max / len(a)):
                return
            norm = scale_a * scale_b * scaled_norm
            from obspy.signal.cross_correlation import correlate
            reverse = correlate(b, a, shift, demean=False,
                                normalize=normalize, method=method)
            output_scale = 1.0 if normalize == "naive" else max(1.0, norm)
            epsilon = _dtype_epsilon(a, b, result)
            tol = 512 * epsilon * max(1, len(a)) * output_scale
            trigger_if(result.shape != reverse.shape or
                       np.any(np.abs(result - reverse[::-1]) > tol),
                       "OB-XCORR-001")

            if normalize == "naive" and _scale(a) < np.finfo(float).max / 4:
                scaled = correlate(2.0 * a, b, shift, demean=False,
                                   normalize=normalize, method=method)
                trigger_if(_different_at_scale(result, scaled, 1.0,
                                               1024 * max(1, len(a))),
                           "OB-XCORR-002")

                direct = correlate(a, b, shift, demean=False,
                                   normalize=normalize, method="direct")
                fft = correlate(a, b, shift, demean=False,
                                normalize=normalize, method="fft")
                trigger_if(_different_at_scale(
                               direct, fft, 1.0, 4096 * max(1, len(a)),
                               epsilon=_dtype_epsilon(a, b, direct, fft)),
                           "OB-XCORR-003")
    except Exception:
        pass
