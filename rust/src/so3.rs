use nalgebra::{Matrix3, Vector3};

pub fn hat(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    )
}

pub fn exp(v: &Vector3<f64>) -> Matrix3<f64> {
    let theta = v.norm();
    if theta < 1e-12 {
        return Matrix3::identity() + hat(v);
    }
    let axis = v / theta;
    let k = hat(&axis);
    Matrix3::identity() + theta.sin() * k + (1.0 - theta.cos()) * (k * k)
}

pub fn log(r: &Matrix3<f64>) -> Vector3<f64> {
    let cos_theta = ((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta.abs() < 1e-12 {
        return Vector3::zeros();
    }
    if (std::f64::consts::PI - theta).abs() < 1e-6 {
        let col = if r[(0, 0)] > r[(1, 1)] && r[(0, 0)] > r[(2, 2)] {
            0
        } else if r[(1, 1)] > r[(2, 2)] {
            1
        } else {
            2
        };
        let mut v = r.column(col) + Vector3::ith(col, 1.0);
        v /= v.norm();
        return v * theta;
    }
    let lnr = (r - r.transpose()) * (theta / (2.0 * theta.sin()));
    Vector3::new(lnr[(2, 1)], lnr[(0, 2)], lnr[(1, 0)])
}

pub fn left_jacobian(v: &Vector3<f64>) -> Matrix3<f64> {
    let theta = v.norm();
    if theta < 1e-12 {
        return Matrix3::identity();
    }
    let axis = v / theta;
    let k = hat(&axis);
    Matrix3::identity()
        + ((1.0 - theta.cos()) / theta) * k
        + (1.0 - theta.sin() / theta) * (k * k)
}

pub fn left_jacobian_inverse(v: &Vector3<f64>) -> Matrix3<f64> {
    let theta = v.norm();
    if theta < 1e-12 {
        return Matrix3::identity();
    }
    let half_theta = theta / 2.0;
    let axis = v / theta;
    let k = hat(&axis);
    Matrix3::identity() - 0.5 * theta * k
        + (1.0 - half_theta * half_theta.cos() / half_theta.sin()) * (k * k)
}

pub fn jr(v: &Vector3<f64>) -> Matrix3<f64> {
    left_jacobian(v).transpose()
}

pub fn jr_inv(v: &Vector3<f64>) -> Matrix3<f64> {
    left_jacobian_inverse(v).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_exp_log_roundtrip() {
        let v = Vector3::new(0.1, 0.2, 0.3);
        let r = exp(&v);
        let v2 = log(&r);
        assert_relative_eq!(v, v2, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_identity() {
        let v = Vector3::zeros();
        let r = exp(&v);
        assert_relative_eq!(r, Matrix3::identity(), epsilon = 1e-10);
    }

    #[test]
    fn test_hat_antisymmetric() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let h = hat(&v);
        assert_relative_eq!(h, -h.transpose(), epsilon = 1e-10);
    }
}
