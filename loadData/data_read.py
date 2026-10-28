import os

import numpy as np
import scipy.io as scio


DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))


# Pavia University uses the ROSIS-03 sensor. The original cube has 115
# nominal bands over 430-860 nm. The commonly used experimental cube keeps
# 103 valid bands after removing original 0-based bad-band indices:
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 114].
PAVIAU_ROSIS03_CENTER_WAVELENGTHS_NM = np.array([
    471.491228, 475.263158, 479.035088, 482.807018, 486.578947,
    490.350877, 494.122807, 497.894737, 501.666667, 505.438596,
    509.210526, 512.982456, 516.754386, 520.526316, 524.298246,
    528.070175, 531.842105, 535.614035, 539.385965, 543.157895,
    546.929825, 550.701754, 554.473684, 558.245614, 562.017544,
    565.789474, 569.561404, 573.333333, 577.105263, 580.877193,
    584.649123, 588.421053, 592.192982, 595.964912, 599.736842,
    603.508772, 607.280702, 611.052632, 614.824561, 618.596491,
    622.368421, 626.140351, 629.912281, 633.684211, 637.45614,
    641.22807, 645.0, 648.77193, 652.54386, 656.315789,
    660.087719, 663.859649, 667.631579, 671.403509, 675.175439,
    678.947368, 682.719298, 686.491228, 690.263158, 694.035088,
    697.807018, 701.578947, 705.350877, 709.122807, 712.894737,
    716.666667, 720.438596, 724.210526, 727.982456, 731.754386,
    735.526316, 739.298246, 743.070175, 746.842105, 750.614035,
    754.385965, 758.157895, 761.929825, 765.701754, 769.473684,
    773.245614, 777.017544, 780.789474, 784.561404, 788.333333,
    792.105263, 795.877193, 799.649123, 803.421053, 807.192982,
    810.964912, 814.736842, 818.508772, 822.280702, 826.052632,
    829.824561, 833.596491, 837.368421, 841.140351, 844.912281,
    848.684211, 852.45614, 856.22807,
], dtype=np.float32)


def _project_path(*parts):
    return os.path.join(DATASET_ROOT, *parts)


def _load_mat(path, key):
    return scio.loadmat(path)[key]


def _load_first_existing_mat(candidates):
    for path, img_key, gt_path, gt_key in candidates:
        if os.path.exists(path) and os.path.exists(gt_path):
            return _load_mat(path, img_key), _load_mat(gt_path, gt_key)
    raise FileNotFoundError(
        "None of the dataset paths exist: {}".format([path for path, _, _, _ in candidates])
    )


def readWavelengths(data_set_name, bands):
    """
    Return center wavelengths for spectral bands.

    Values are in nanometers when wavelength metadata is known. FWHM is not
    available for the current metadata, so this returns None for fwhm.
    """
    normalized_name = data_set_name.lower()
    if normalized_name == "paviau":
        if bands != PAVIAU_ROSIS03_CENTER_WAVELENGTHS_NM.size:
            raise ValueError(
                "PaviaU ROSIS-03 wavelength metadata expects 103 valid bands "
                "after removing bad bands [0,1,2,3,4,5,6,7,8,9,10,114], "
                "but got {} bands.".format(bands)
            )
        return PAVIAU_ROSIS03_CENTER_WAVELENGTHS_NM.copy(), None

    known_ranges_nm = {
        # Indian Pines corrected data keeps AVIRIS bands after noisy bands removal.
        # This is an approximate range; exact band centers should replace it if available.
        "indian": (400.0, 2500.0),
        # The bundled Xuzhou readme states that 436 retained bands cover 415-2508 nm.
        "xuzhou": (415.0, 2508.0),
        # WHU-Hi scenes are visible/NIR cubes. Replace this with exact band centers
        # if a sensor metadata file becomes available.
        "whu": (400.0, 1000.0),
        "whuh": (400.0, 1000.0),
        "whu-honghu": (400.0, 1000.0),
        "whu_honghu": (400.0, 1000.0),
        "whul": (400.0, 1000.0),
        "whu-longkou": (400.0, 1000.0),
        "whu_longkou": (400.0, 1000.0),
        "whuhc": (400.0, 1000.0),
        "whu-hanchuan": (400.0, 1000.0),
        "whu_hanchuan": (400.0, 1000.0),
    }
    if normalized_name in known_ranges_nm:
        start_nm, end_nm = known_ranges_nm[normalized_name]
        return np.linspace(start_nm, end_nm, bands, dtype=np.float32), None

    return np.linspace(0.0, 1.0, bands, dtype=np.float32), None


def readData(data_set_name):
    data_key = data_set_name.lower()

    if data_key == "indian":
        return _load_first_existing_mat([
            (_project_path("Indian", "Indian_pines_corrected.mat"), "indian_pines_corrected",
             _project_path("Indian", "Indian_pines_gt.mat"), "indian_pines_gt"),
            (r"D:\dataset\Indian\Indian_pines_corrected.mat", "indian_pines_corrected",
             r"D:\dataset\Indian\Indian_pines_gt.mat", "indian_pines_gt"),
        ])

    if data_key == "paviau":
        return _load_first_existing_mat([
            (_project_path("PaviaU", "PaviaU.mat"), "paviaU",
             _project_path("PaviaU", "PaviaU_gt.mat"), "paviaU_gt"),
            (r"C:\Users\remote\Desktop\hyper-spectr\dataset\PaviaU\PaviaU.mat", "paviaU",
             r"C:\Users\remote\Desktop\hyper-spectr\dataset\PaviaU\PaviaU_gt.mat", "paviaU_gt"),
        ])

    if data_key == "xuzhou":
        return _load_first_existing_mat([
            (_project_path("Xuzhou", "xuzhou.mat"), "xuzhou",
             _project_path("Xuzhou", "xuzhou_gt.mat"), "xuzhou_gt"),
            (r"D:\dataset\Xuzhou\xuzhou.mat", "xuzhou",
             r"D:\dataset\Xuzhou\xuzhou_gt.mat", "xuzhou_gt"),
        ])

    if data_key in ["whu", "whuh", "whu-honghu", "whu_honghu"]:
        img, gt = _load_first_existing_mat([
            (_project_path("WHU-Hi", "WHU-Hi-HongHu", "WHU_Hi_HongHu.mat"), "WHU_Hi_HongHu",
             _project_path("WHU-Hi", "WHU-Hi-HongHu", "WHU_Hi_HongHu_gt.mat"), "WHU_Hi_HongHu_gt"),
            (r"D:\dataset\WHU-Hi-HongHu\WHU_Hi_HongHu.mat", "WHU_Hi_HongHu",
             r"D:\dataset\WHU-Hi-HongHu\WHU_Hi_HongHu_gt.mat", "WHU_Hi_HongHu_gt"),
        ])
        if data_set_name == "WHU":
            img = img[450:, :, :]
            gt = gt[450:, :]
            gt[gt > 5] -= 1
        return img, gt

    if data_key in ["whul", "whu-longkou", "whu_longkou"]:
        return _load_first_existing_mat([
            (_project_path("WHU-Hi", "WHU-Hi-LongKou", "WHU_Hi_LongKou.mat"), "WHU_Hi_LongKou",
             _project_path("WHU-Hi", "WHU-Hi-LongKou", "WHU_Hi_LongKou_gt.mat"), "WHU_Hi_LongKou_gt"),
            (r"D:\dataset\WHU-Hi-LongKou\WHU_Hi_LongKou.mat", "WHU_Hi_LongKou",
             r"D:\dataset\WHU-Hi-LongKou\WHU_Hi_LongKou_gt.mat", "WHU_Hi_LongKou_gt"),
        ])

    if data_key in ["whuhc", "whu-hanchuan", "whu_hanchuan"]:
        return _load_first_existing_mat([
            (_project_path("WHU-Hi", "WHU-Hi-HanChuan", "WHU_Hi_HanChuan.mat"), "WHU_Hi_HanChuan",
             _project_path("WHU-Hi", "WHU-Hi-HanChuan", "WHU_Hi_HanChuan_gt.mat"), "WHU_Hi_HanChuan_gt"),
        ])

    if data_key in ["laoyuhe", "laoyuhesidi", "lao-yu-he-sidi"]:
        img_path = _project_path("LaoYuHeSidi", "LapYuHeSiDi.npy")
        gt_path = _project_path("LaoYuHeSidi", "LapYuHeSiDi_label.npy")
        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            raise FileNotFoundError("LaoYuHeSidi npy files were not found under {}".format(os.path.dirname(img_path)))
        return np.load(img_path), np.load(gt_path)

    if data_set_name == "salinas":
        img = scio.loadmat(r"D:\dataset\Salinas\salinas.mat")["HSI_original"]
        gt = scio.loadmat(r"D:\dataset\Salinas\salinas_gt.mat")["Data_gt"]
        return img, gt

    if data_set_name == "KSC":
        img = scio.loadmat(r"D:\dataset\KSC\KSC.mat")["KSC"]
        gt = scio.loadmat(r"D:\dataset\KSC\KSC_gt.mat")["KSC_gt"]
        return img, gt

    if data_set_name in ["HoustonU", "xiongan"]:
        import h5py

        if data_set_name == "HoustonU":
            img_file = h5py.File(r"D:\dataset\HoustonU\HoustonU.mat", "r")
            img = np.array(img_file.get("houstonU"))
            img = np.transpose(img, (1, 2, 0))
            gt_file = h5py.File(r"D:\dataset\HoustonU\HoustonU_gt.mat", "r")
            gt = np.array(gt_file.get("houstonU_gt"))
            return img, gt

        img_file = h5py.File(r"D:\dataset\XiongAn\xiongan.mat", "r")
        img = np.array(img_file.get("XiongAn"))
        img = np.transpose(img, (1, 2, 0))
        gt_file = h5py.File(r"D:\dataset\XiongAn\xiongan_gt.mat", "r")
        gt = np.array(gt_file.get("xiongan_gt"))
        return img, gt

    if data_set_name == "Houston2013":
        img = scio.loadmat(r"D:\dataset\HoustonU\Houston.mat")["Houston"]
        gt = scio.loadmat(r"D:\dataset\HoustonU\Houston_gt.mat")["Houston_gt"]
        return img, gt

    if data_set_name == "DC":
        img = np.load(r"D:\dataset\DianChi\DCdatasetsmall\HSIDC.npy")
        gt = np.load(r"D:\dataset\DianChi\DCdatasetsmall\LabelDC.npy")
        return img, gt

    raise ValueError("Unknown dataset name: {}".format(data_set_name))
