import numpy as np
from numba import jit, njit

from config import *


@njit
def pairwise_distance_numba(pos_array1, pos_array2):
    """Calculates pairwise Euclidean distances between two sets of 3D points."""
    num_pos1 = pos_array1.shape[0]
    num_pos2 = pos_array2.shape[0]
    distances = np.empty((num_pos1, num_pos2))
    for i in range(num_pos1):
        for j in range(num_pos2):
            dx = pos_array1[i, 0] - pos_array2[j, 0]
            dy = pos_array1[i, 1] - pos_array2[j, 1]
            dz = pos_array1[i, 2] - pos_array2[j, 2]
            distances[i, j] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return distances


@jit(nopython=True)
def _calculate_datarate_numba(bw_hz, p_watt, pl_db, noise_const):
    """Shannon-Hartley theorem calculation."""
    noise_watt = noise_const * bw_hz
    rx_watt_dbm = 10 * np.log10(p_watt * 1000) - pl_db
    rx_watt = 10 ** ((rx_watt_dbm - 30) / 10)
    snr = rx_watt / noise_watt
    return (bw_hz * np.log2(1 + snr)) if snr > 0 else 0.0


class CommunicationModel:
    def __init__(self):
        self.noise_const = 10 ** ((NOISE_POWER_SPECTRAL_DENSITY - 30) / 10)

    def get_distance(self, e1, e2):
        return np.linalg.norm(e1.position - e2.position)

    def calculate_los_probability(self, uav, entity):
        dist_2d = np.linalg.norm(uav.position[:2] - entity.position[:2])
        delta_h = abs(uav.position[2] - entity.position[2])
        angle_deg = np.rad2deg(np.arctan(delta_h / dist_2d) if dist_2d > 0 else np.pi / 2)
        return 1 / (1 + LOS_X0 * np.exp(-LOS_Y0 * (angle_deg - LOS_X0)))

    def calculate_path_loss(self, dist, is_los):
        if dist == 0: return 0
        fspl = 20 * np.log10(dist) + 20 * np.log10(CARRIER_FREQUENCY) - 147.55
        return fspl + (ETA_LOS if is_los else ETA_NLOS)

    def get_average_path_loss(self, e1, e2):
        dist = self.get_distance(e1, e2)
        if dist == 0: return 0

        los_prob = self.calculate_los_probability(e1, e2)
        pl_los = self.calculate_path_loss(dist, True)
        pl_nlos = self.calculate_path_loss(dist, False)

        # Probabilistic average path loss
        avg_pl_linear = los_prob * (10 ** (pl_los / 10)) + (1 - los_prob) * (10 ** (pl_nlos / 10))
        return 10 * np.log10(avg_pl_linear)

    def compute_rate_uav_user(self, user, uav, num_sharing_users=1):
        bw = BANDWIDTH_UAV_USER / num_sharing_users if DYNAMIC_BANDWIDTH else BANDWIDTH_UAV_USER
        pl_db = self.get_average_path_loss(uav, user)
        return _calculate_datarate_numba(bw, POWER_UAV_USER, pl_db, self.noise_const) / 1e6  # Mbps

    def compute_rate_uav_uav(self, uav1, uav2):
        pl_db = self.get_average_path_loss(uav1, uav2)
        return _calculate_datarate_numba(BANDWIDTH_UAV_UAV, POWER_UAV_UAV, pl_db, self.noise_const) / 1e6

    def compute_rate_uav_cloud(self, uav):
        # Simplified fixed path loss to cloud for now
        return _calculate_datarate_numba(BANDWIDTH_UAV_CCC, POWER_CCC, 60, self.noise_const) / 1e6
