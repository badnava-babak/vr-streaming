from __future__ import annotations

import numpy as np
import pandas as pd


class User:
    def __init__(self, head_navigation_df: pd.DataFrame):
        self.head_navigation_df = head_navigation_df  # (yaw, pitch, roll, time stamp [sec])

    def get_viewport(self, t: int) -> np.ndarray:
        yaw, pitch, roll, _ = self.head_navigation_df[t]
        viewport = self._visible_tiles(yaw, pitch, roll)
        return viewport

    def _visible_tiles(self,
                       yaw_deg: float,
                       pitch_deg: float,
                       roll_deg: float,
                       *,
                       fov_h_deg: float = 90,  # horizontal FOV
                       fov_v_deg: float = 90,  # vertical   FOV
                       tiles_lat: int = 8,  # rows   (south→north)
                       tiles_lon: int = 8  # cols   (west→east)
                       ) -> np.ndarray:
        """
        Return a (tiles_lat × tiles_lon) Boolean array whose True entries
        are the tiles that lie inside the viewer's rectangular FOV.

        Coordinate system
        -----------------
        • World forward  = –z,   right  = +x,   up = +y   (OpenGL / WebXR)
        • yaw    (deg)  : rotation around +y   (heading, east-positive)
        • pitch  (deg)  : rotation around +x   (look up = +, down = –)
        • roll   (deg)  : rotation around +z   (tilt clockwise = +)
        """

        # -------------------------------------------------------------
        # 1. rotation matrix  R_world→camera  (yaw•pitch•roll extrinsic)
        # -------------------------------------------------------------
        yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])

        R_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                          [0, 1, 0],
                          [-np.sin(yaw), 0, np.cos(yaw)]])
        R_pitch = np.array([[1, 0, 0],
                            [0, np.cos(pitch), -np.sin(pitch)],
                            [0, np.sin(pitch), np.cos(pitch)]])
        R_roll = np.array([[np.cos(roll), -np.sin(roll), 0],
                           [np.sin(roll), np.cos(roll), 0],
                           [0, 0, 1]])

        R = R_roll @ R_pitch @ R_yaw  # world → camera
        Rt = R.T  # camera → world  (for later)

        # -------------------------------------------------------------
        # 2. unit-vector centre of every tile in world coordinates
        # -------------------------------------------------------------
        lats = np.linspace(-90, 90, tiles_lat, endpoint=False) + 90 / tiles_lat
        lons = np.linspace(-180, 180, tiles_lon, endpoint=False) + 180 / tiles_lon
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')  # shape (L, W)

        lat_r, lon_r = np.deg2rad(lat_grid), np.deg2rad(lon_grid)
        # spherical → Cartesian
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.sin(lat_r)
        z = np.cos(lat_r) * np.sin(lon_r)
        dirs_world = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (T,3)

        # -------------------------------------------------------------
        # 3. express those directions in the *camera* frame
        # -------------------------------------------------------------
        dirs_cam = dirs_world @ R  # (T,3)

        # forward = –z in camera space; horizontal angle = atan2(x,-z)
        h = np.arctan2(dirs_cam[:, 0], -dirs_cam[:, 2]) * 180 / np.pi  # deg
        v = np.arctan2(dirs_cam[:, 1], -dirs_cam[:, 2]) * 180 / np.pi

        mask = (np.abs(h) <= fov_h_deg / 2) & (np.abs(v) <= fov_v_deg / 2)
        return mask.reshape(tiles_lat, tiles_lon)
