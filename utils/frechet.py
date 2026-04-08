import math
import torch


_PADE13_CONSTS_CACHE = {}



def mat_norm1(A: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.sum(torch.abs(A), dim=0))



def _pade_b(m: int, dtype, device):
    two_m_fact = math.factorial(2 * m)
    m_fact = math.factorial(m)
    b = []
    for k in range(m + 1):
        num = math.factorial(2 * m - k) * m_fact
        den = two_m_fact * math.factorial(m - k) * math.factorial(k)
        b.append(num / den)
    return torch.tensor(b, dtype=dtype, device=device)



def pade_uv(A: torch.Tensor, m: int, I: torch.Tensor):
    b = _pade_b(m, A.dtype, A.device)
    A2 = A @ A
    Apow = [None] * m
    Apow[1] = A2
    for p in range(2, m):
        Apow[p] = Apow[p - 1] @ A2

    V = b[0] * I
    for k in range(2, m + 1, 2):
        p = k // 2
        V = V + b[k] * Apow[p]

    W = b[1] * I
    for k in range(3, m + 1, 2):
        p = (k - 1) // 2
        W = W + b[k] * Apow[p]

    U = A @ W
    return U, V



def pade_frechet_LuLv(A: torch.Tensor, E: torch.Tensor, m: int, I: torch.Tensor):
    b = _pade_b(m, A.dtype, A.device)
    A2 = A @ A
    M2 = A @ E + E @ A

    Apow = [None] * m
    Mpow = [None] * m
    Apow[1] = A2
    Mpow[1] = M2

    for p in range(2, m):
        Apow[p] = Apow[p - 1] @ A2
        Mpow[p] = Apow[p - 1] @ M2 + Mpow[p - 1] @ A2

    Lv = torch.zeros_like(A)
    for k in range(2, m + 1, 2):
        p = k // 2
        Lv = Lv + b[k] * Mpow[p]

    W = b[1] * I
    for k in range(3, m + 1, 2):
        p = (k - 1) // 2
        W = W + b[k] * Apow[p]

    Lw = torch.zeros_like(A)
    for k in range(3, m + 1, 2):
        p = (k - 1) // 2
        Lw = Lw + b[k] * Mpow[p]

    Lu = A @ Lw + E @ W
    return Lu, Lv



def expm_frechet_alg64_pade13(A: torch.Tensor, E: torch.Tensor):
    assert A.shape == E.shape and A.ndim == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    dev = A.device
    dtype = A.dtype

    b = torch.tensor(
        [
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ],
        device=dev,
        dtype=dtype,
    )

    theta_13 = torch.tensor(5.371920351148152, device=dev, dtype=dtype)
    I = torch.eye(n, device=dev, dtype=dtype)
    A_norm1 = mat_norm1(A)

    ell3 = 1.495585217958292e-02
    ell5 = 2.539398330063230e-01
    ell7 = 9.504178996162932e-01
    ell9 = 2.097847961257068e00

    m = None
    if A_norm1 <= ell3:
        m = 3
    elif A_norm1 <= ell5:
        m = 5
    elif A_norm1 <= ell7:
        m = 7
    elif A_norm1 <= ell9:
        m = 9

    if m is not None:
        U, V = pade_uv(A, m, I)
        Lu, Lv = pade_frechet_LuLv(A, E, m, I)
        K = V - U
        LU, piv = torch.linalg.lu_factor(K)
        R = torch.linalg.lu_solve(LU, piv, V + U)
        rhs_L = (Lu + Lv) + (Lu - Lv) @ R
        L = torch.linalg.lu_solve(LU, piv, rhs_L)
        return R, L

    if A_norm1 <= theta_13:
        s = 0
        A_s = A
        E_s = E
    else:
        s = int(torch.ceil(torch.log2(A_norm1 / theta_13)).item())
        scale = 2.0 ** (-s)
        A_s = A * scale
        E_s = E * scale

    A2 = A_s @ A_s
    A4 = A2 @ A2
    A6 = A2 @ A4

    M2 = A_s @ E_s + E_s @ A_s
    M4 = A2 @ M2 + M2 @ A2
    M6 = A4 @ M2 + M4 @ A2

    W1 = b[13] * A6 + b[11] * A4 + b[9] * A2
    W2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I
    Z1 = b[12] * A6 + b[10] * A4 + b[8] * A2
    Z2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I

    W = A6 @ W1 + W2
    U = A_s @ W
    V = A6 @ Z1 + Z2

    Lw1 = b[13] * M6 + b[11] * M4 + b[9] * M2
    Lw2 = b[7] * M6 + b[5] * M4 + b[3] * M2
    Lz1 = b[12] * M6 + b[10] * M4 + b[8] * M2
    Lz2 = b[6] * M6 + b[4] * M4 + b[2] * M2

    Lw = A6 @ Lw1 + M6 @ W1 + Lw2
    Lu = A_s @ Lw + E_s @ W
    Lv = A6 @ Lz1 + M6 @ Z1 + Lz2

    K = V - U
    LU, piv = torch.linalg.lu_factor(K)
    R = torch.linalg.lu_solve(LU, piv, V + U)
    rhs_L = (Lu + Lv) + (Lu - Lv) @ R
    L = torch.linalg.lu_solve(LU, piv, rhs_L)

    for _ in range(s):
        L = R @ L + L @ R
        R = R @ R

    return R, L



def _get_pade13_consts(n: int, device: torch.device, dtype: torch.dtype):
    key = (n, str(device), str(dtype))
    if key not in _PADE13_CONSTS_CACHE:
        b = torch.tensor(
            [
                64764752532480000.0,
                32382376266240000.0,
                7771770303897600.0,
                1187353796428800.0,
                129060195264000.0,
                10559470521600.0,
                670442572800.0,
                33522128640.0,
                1323241920.0,
                40840800.0,
                960960.0,
                16380.0,
                182.0,
                1.0,
            ],
            device=device,
            dtype=dtype,
        )
        theta_13 = torch.tensor(5.371920351148152, device=device, dtype=dtype)
        I = torch.eye(n, device=device, dtype=dtype)
        _PADE13_CONSTS_CACHE[key] = (b, theta_13, I)
    return _PADE13_CONSTS_CACHE[key]



def expm_pade13_prepare(A: torch.Tensor):
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    dev = A.device
    dtype = A.dtype

    b, theta_13, I = _get_pade13_consts(n, dev, dtype)
    A_norm1 = mat_norm1(A)

    ell3 = 1.495585217958292e-02
    ell5 = 2.539398330063230e-01
    ell7 = 9.504178996162932e-01
    ell9 = 2.097847961257068e00

    m = None
    if A_norm1 <= ell3:
        m = 3
    elif A_norm1 <= ell5:
        m = 5
    elif A_norm1 <= ell7:
        m = 7
    elif A_norm1 <= ell9:
        m = 9

    if m is not None:
        U, V = pade_uv(A, m, I)
        K = V - U
        LU, piv = torch.linalg.lu_factor(K)
        R = torch.linalg.lu_solve(LU, piv, V + U)
        cache = {"branch": "small", "m": m, "A": A, "I": I, "R": R, "LU": LU, "piv": piv}
        return R, cache

    if A_norm1 <= theta_13:
        s = 0
        scale = 1.0
        A_s = A
    else:
        s = int(torch.ceil(torch.log2(A_norm1 / theta_13)).item())
        scale = 2.0 ** (-s)
        A_s = A * scale

    A2 = A_s @ A_s
    A4 = A2 @ A2
    A6 = A2 @ A4

    W1 = b[13] * A6 + b[11] * A4 + b[9] * A2
    W2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I
    Z1 = b[12] * A6 + b[10] * A4 + b[8] * A2
    Z2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I

    W = A6 @ W1 + W2
    U = A_s @ W
    V = A6 @ Z1 + Z2

    K = V - U
    LU, piv = torch.linalg.lu_factor(K)
    R0 = torch.linalg.lu_solve(LU, piv, V + U)

    R_steps = []
    R = R0
    for _ in range(s):
        R_steps.append(R)
        R = R @ R
    R_final = R

    cache = {
        "branch": "pade13",
        "b": b,
        "I": I,
        "s": s,
        "scale": scale,
        "A_s": A_s,
        "A2": A2,
        "A4": A4,
        "A6": A6,
        "W": W,
        "W1": W1,
        "Z1": Z1,
        "LU": LU,
        "piv": piv,
        "R0": R0,
        "R_steps": R_steps,
    }
    return R_final, cache



def expm_frechet_from_cache(cache: dict, E: torch.Tensor):
    if cache["branch"] == "small":
        m = cache["m"]
        A = cache["A"]
        I = cache["I"]
        R = cache["R"]
        LU = cache["LU"]
        piv = cache["piv"]

        Lu, Lv = pade_frechet_LuLv(A, E, m, I)
        rhs_L = (Lu + Lv) + (Lu - Lv) @ R
        return torch.linalg.lu_solve(LU, piv, rhs_L)

    b = cache["b"]
    s = cache["s"]
    scale = cache["scale"]
    A_s = cache["A_s"]
    A2 = cache["A2"]
    A4 = cache["A4"]
    A6 = cache["A6"]
    W = cache["W"]
    W1 = cache["W1"]
    Z1 = cache["Z1"]
    LU = cache["LU"]
    piv = cache["piv"]
    R0 = cache["R0"]
    R_steps = cache["R_steps"]

    E_s = E * scale
    M2 = A_s @ E_s + E_s @ A_s
    M4 = A2 @ M2 + M2 @ A2
    M6 = A4 @ M2 + M4 @ A2

    Lw1 = b[13] * M6 + b[11] * M4 + b[9] * M2
    Lw2 = b[7] * M6 + b[5] * M4 + b[3] * M2
    Lz1 = b[12] * M6 + b[10] * M4 + b[8] * M2
    Lz2 = b[6] * M6 + b[4] * M4 + b[2] * M2

    Lw = A6 @ Lw1 + M6 @ W1 + Lw2
    Lu = A_s @ Lw + E_s @ W
    Lv = A6 @ Lz1 + M6 @ Z1 + Lz2

    rhs_L = (Lu + Lv) + (Lu - Lv) @ R0
    L = torch.linalg.lu_solve(LU, piv, rhs_L)

    for Rk in R_steps:
        L = Rk @ L + L @ Rk
    return L
