
COVERAGE = {
    4: (0.6221, 0.8256),
    5: (0.6802, 0.8779),
    6: (0.7326, 0.9186),
    7: (0.7674, 0.9477),
    8: (0.7907, 0.9593),
    9: (0.8023, 0.9761),
}

DIESEL_COST_PER_KM = 54.20
COST_GAP_AT_4_STATIONS = {
    "plan1_non_flexible": 12.3,
    "plan2_flexible_swap": 15.7,
    "plan3_flexible_charge": 19.9,
}

TIME_P75_INCREASE_PLAN2_8_STATIONS = 0.065
TIME_P75_INCREASE_RANGE = (0.01, 0.079)

INVENTORY_TABLE3 = {
    "Nanjing":      dict(initial=(567, 338), average=(104, 101), maximum=(301, 181), std=(39.4, 55.7)),
    "Changshu":     dict(initial=(34, 36),   average=(26, 30),   maximum=(34, 47),   std=(8.7, 9.4)),
    "Jiangyin":     dict(initial=(97, 51),   average=(43, 21),   maximum=(85, 39),   std=(19.7, 11.3)),
    "Wuhu":         dict(initial=(269, 193), average=(48, 32),   maximum=(130, 74),  std=(18.1, 13.3)),
    "Yangzhou":     dict(initial=(149, 69),  average=(60, 49),   maximum=(79, 113),  std=(16.0, 31.5)),
    "Taicang":      dict(initial=(388, 378), average=(78, 73),   maximum=(240, 246), std=(38.1, 49.6)),
    "Yangshan":     dict(initial=(906, 835), average=(84, 118),  maximum=(368, 366), std=(52.9, 69.4)),
    "Zhangjiagang": dict(initial=(85, 27),   average=(38, 41),   maximum=(66, 94),   std=(10.8, 28.2)),
}

N_SHIPS = 172
N_PORTS = 19
N_SEGMENTS = 4360

ABSTRACT_COVERAGE_CLAIM = 0.98
ABSTRACT_CHARGING_CLAIM = 0.80
