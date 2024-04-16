import torch

def lyapunov(P: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    n = z.shape[0] // 2
    u, v = z[:n], z[n:]
    return -torch.sum(P) + torch.sum(u) + torch.sum(v)


def lyapunov_grad(P: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    return torch.cat([1 - torch.sum(P, 0), 1 - torch.sum(P, 1)], dim=0)


def lyapunov_hessian(P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row_sum = torch.sum(P, 0)
    col_sum = torch.sum(P, 1)

    row1 = torch.cat([torch.diag(row_sum), P], dim=1)
    row2 = torch.cat([P.T, torch.diag(col_sum)], dim=1)

    return torch.cat([row1, row2], dim=0)


def conjgrad(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x = torch.randn_like(b)
    r = b - A @ x
    p = r
    r_squared_old = torch.dot(r, r)
    for _ in b:
        Ap = A @ p
        alpha = r_squared_old / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        r_squared_new = torch.dot(r, r)
        if torch.sqrt(r_squared_new) < 1e-8:
            break
        p = r + (r_squared_new/r_squared_old)*p
        r_squared_old = r_squared_new
    return x


def backtrack(func, grad_func, x, p, tau=0.5, alpha=1.0, c1=1e-3, max_iter=100):
    phi0, grad0 = func(x), grad_func(x)
    dphi0 = torch.dot(grad0, p)

    if dphi0 >= 0.0:
        return ValueError('Must provide a descent direction')

    for _ in range(max_iter):
        if torch.all(func(x + alpha*p) < phi0 + c1*alpha*dphi0):
            return alpha
        alpha *= tau

    return None


def newton(C: torch.Tensor, u, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The newton stage. (Without using sparse hessian). """

    n = len(u)

    for _ in range(2):
        P = torch.exp(-C + u[:, None] + v[None, :] - 1)
        M = lyapunov_hessian(P)
        z = torch.cat([u, v], dim=0)
        grad = conjgrad(M, -lyapunov_grad(P, z))
        alpha = backtrack(partial(lyapunov, P),
                          partial(lyapunov_grad, P), z, grad)
        assert not alpha is None
        u += alpha * grad[:n]
        v += alpha * grad[n:]

    return u, v
