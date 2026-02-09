import numpy as np
import warnings


def to_polar(x, y, z, units="degrees"):
    """
    convert arrays of [x,y,z] points to arrays of [phi,theta,r] in degrees
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta = np.arccos(-z / r)

    theta = np.nan_to_num(theta, nan=0)
    phi = np.arctan2(x, y)
    phi[np.where(phi < 0)] = phi[np.where(phi < 0)] + 2 * np.pi

    if units == "degrees":
        theta = np.degrees(theta)
        phi = np.degrees(phi)

    return np.array((theta, phi, r))


def to_cartesian(theta, phi, r):
    """
    convert from degrees polar to cartesian coordinates
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    x = r * np.sin(theta_rad) * np.sin(phi_rad)
    y = r * np.sin(theta_rad) * np.cos(phi_rad)
    z = r * np.cos(theta_rad)

    return np.array((x, y, z))


def attitude(coords, roll, pitch, yaw):

    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    R_roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )

    R_pitch = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    R_yaw = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    # Apply rotations
    coordst = coords.copy()
    coordst = np.dot(R_roll, coordst)
    coordst = np.dot(R_pitch, coordst)
    coordst = np.dot(R_yaw, coordst)
    newx, newy, newz = coordst

    return newx, newy, newz
