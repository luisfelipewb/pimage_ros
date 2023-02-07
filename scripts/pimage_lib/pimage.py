from typing import List
import cv2
import numpy as np

def demosaicing(img_raw: np.ndarray) -> List[np.ndarray]:
    """Polarization demosaicing
    Parameters
    ----------
    img_raw : np.ndarray
        Polarization image taken with polarizatin sensor
    Returns
    -------
    img_demosaiced_list : List[np.ndarray]
        List of demosaiced images. The shape of each image is (height, width, 3).
    """
    # split
    # (0, 0):90,  (0, 1):45 (1, 0):135, (1, 1):0
    img_bayer_090 = img_raw[0::2, 0::2]
    img_bayer_045 = img_raw[0::2, 1::2]
    img_bayer_135 = img_raw[1::2, 0::2]
    img_bayer_000 = img_raw[1::2, 1::2]

    # debayer
    img_bgr_090 = cv2.cvtColor(img_bayer_090, cv2.COLOR_BayerBG2BGR)
    img_bgr_045 = cv2.cvtColor(img_bayer_045, cv2.COLOR_BayerBG2BGR)
    img_bgr_135 = cv2.cvtColor(img_bayer_135, cv2.COLOR_BayerBG2BGR)
    img_bgr_000 = cv2.cvtColor(img_bayer_000, cv2.COLOR_BayerBG2BGR)

    return [img_bgr_000, img_bgr_045, img_bgr_090, img_bgr_135]

def rgb(demosaiced_list: List[np.ndarray]) -> np.ndarray:
    """Extract rgb image
    Parameters
    ----------
    demosaiced_list : List of demosaiced images.
    Returns
    -------
    img_bgr : Collored image
    """
    img_a = cv2.addWeighted(demosaiced_list[0], 0.5, demosaiced_list[2], 0.5, 0.0)
    img_b = cv2.addWeighted(demosaiced_list[1], 0.5, demosaiced_list[3], 0.5, 0.0)
    img_bgr = cv2.addWeighted(img_a, 0.5, img_b, 0.5, 0.0)

    return img_bgr


def calcStokes(demosaiced_list: List[np.ndarray]) -> np.ndarray:
    """ Compute stokes vector
    Parameters
    ----------
    demosaiced_list : List of demosaiced images. (assuming order of 0, 45, 90, 135)
    Returns
    -------
    stokes : stokes vector
    """
    s0 = np.sum(demosaiced_list, axis=(0), dtype=np.float)/2
    s1 = demosaiced_list[0].astype(np.float) - demosaiced_list[2].astype(np.float) #0-90
    s2 = demosaiced_list[1].astype(np.float) - demosaiced_list[3].astype(np.float) #45-135
    print(f"input shape {demosaiced_list[0].dtype}")
    print(f"Shape: {np.shape(s0)} {s0.dtype}") 
    print(f"Shape: {np.shape(s1)} {s1.dtype}") 
    print(f"Shape: {np.shape(s2)} {s1.dtype}")
    return np.stack((s0, s1, s2))


def calcDoLP(stokes):
    # return np.sqrt(s1**2 + s2**2) / s0
    return np.sqrt(stokes[1]**2 + stokes[2]**2) / stokes[0]