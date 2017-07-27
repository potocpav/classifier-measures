 #![warn(missing_docs)]

 /*!
Measure classifier's performance using [Receiver Operating
Characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
(ROC) and [Precision-Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
(PR) curves.

The curves themselves can be computed as well as trapezoidal areas under the curves.
 */

extern crate num_traits;
use num_traits::{Float, NumCast};

mod test;

/// Integration using the trapezoidal rule.
fn trapezoidal<F: Float>(x: &[F], y: &[F]) -> F {
    let mut prev_x = x[0];
    let mut prev_y = y[0];
    let mut integral = F::zero();

    for (&x, &y) in x.iter().skip(1).zip(y.iter().skip(1)) {
        integral = integral + (x - prev_x) * (prev_y + y) / NumCast::from(2.0).unwrap();
        prev_x = x;
        prev_y = y;
    }
    integral
}

/// Checks if all data-points are finite
fn check_data<F: Float>(v: &[(bool, F)]) -> bool {
    v.iter().any(|x| x.0) && v.iter().any(|x| !x.0) && v.iter().all(|x| x.1.is_finite())
}

/// Uses a provided closure to convert free-form data into a standard format
fn convert<T, X, F, I>(data: I, convert_fn: F) -> Vec<(bool, X)> where
        I: IntoIterator<Item=T>,
        F: Fn(T) -> (bool, X),
        X: Float {
    let data_it = data.into_iter();
    let mut v = Vec::with_capacity(data_it.size_hint().0);
    for i in data_it {
        v.push(convert_fn(i));
    }
    v
}

/// Computes a ROC curve of a given classifier sorting `pairs` in-place.
///
/// Returns `None` if one of the classes is not present or any values are non-finite.
/// Otherwise, returns `Some((v_x, v_y))` where `v_x` are the x-coordinates and `v_y` are the
/// y-coordinates of the ROC curve.
pub fn roc_mut<F: Float>(pairs: &mut [(bool, F)]) -> Option<(Vec<F>, Vec<F>)> {
    if !check_data(pairs) {
        return None;
    }
    pairs.sort_by(&|x: &(_, F), y: &(_, F)|
        match y.1.partial_cmp(&x.1) {
            Some(ord) => ord,
            None      => unreachable!(),
        });

    let mut s0 = F::nan();
    let (mut tp, mut fp) = (F::zero(), F::zero());
    let (mut tps, mut fps) = (vec![], vec![]);
    for &(t, s) in pairs.iter() {
        if s != s0 {
            tps.push(tp);
            fps.push(fp);
            s0 = s;
        }
        match t {
            false => fp = fp + F::one(),
            true =>  tp = tp + F::one(),
        }
    }
    tps.push(tp);
    fps.push(fp);

    // normalize
    let (tp_max, fp_max) = (tps[tps.len() - 1], fps[fps.len() - 1]);
    for mut tp in &mut tps {
        *tp = *tp / tp_max;
    }
    for mut fp in &mut fps {
        *fp = *fp / fp_max;
    }
    Some((fps, tps))
}

/// Computes a ROC curve of a given classifier.
///
/// `data` is a free-form `IntoIterator` object and `convert_fn` is a closure that converts each
/// data-point into a pair `(ground_truth, prediction).`
///
/// Returns `None` if one of the classes is not present or any values are non-finite.
/// Otherwise, returns `Some((v_x, v_y))` where `v_x` are the x-coordinates and `v_y` are the
/// y-coordinates of the ROC curve.
pub fn roc<T, X, F, I>(data: I, convert_fn: F) -> Option<(Vec<X>, Vec<X>)> where
        I: IntoIterator<Item=T>,
        F: Fn(T) -> (bool, X),
        X: Float {
    roc_mut(&mut convert(data, convert_fn))
}

/// Computes a PR curve of a given classifier.
///
/// `data` is a free-form `IntoIterator` object and `convert_fn` is a closure that converts each
/// data-point into a pair `(ground_truth, prediction).`
///
/// Returns `None` if one of the classes is not present or any values are non-finite.
/// Otherwise, returns `Some((v_x, v_y))` where `v_x` are the x-coordinates and `v_y` are the
/// y-coordinates of the PR curve.
pub fn pr<T, X, F, I>(data: I, convert_fn: F) -> Option<(Vec<X>, Vec<X>)> where
        I: IntoIterator<Item=T>,
        F: Fn(T) -> (bool, X),
        X: Float {
    pr_mut(&mut convert(data, convert_fn))
}

/// Computes a PR curve of a given classifier sorting `pairs` in-place.
///
/// Returns `None` if one of the classes is not present or any values are non-finite.
/// Otherwise, returns `Some((v_x, v_y))` where `v_x` are the x-coordinates and `v_y` are the
/// y-coordinates of the PR curve.
pub fn pr_mut<F: Float>(pairs: &mut [(bool, F)]) -> Option<(Vec<F>, Vec<F>)> {
    if !check_data(pairs) {
        return None;
    }
    pairs.sort_by(&|x: &(_, F), y: &(_, F)|
        match y.1.partial_cmp(&x.1) {
            Some(ord) => ord,
            None      => unreachable!(),
        });

    let mut x0 = F::nan();
    let (mut tp, mut p, mut fp) = (F::zero(), F::zero(), F::zero());
    let (mut recall, mut precision) = (vec![], vec![]);

    // number of labels
    let ln = pairs.iter().fold(F::zero(), |a,b| a + if b.0 { F::one() } else { F::zero() });

    for &(l, x) in pairs.iter() {
        if x != x0 {
            recall.push(tp / ln);
            precision.push(if p == F::zero() { F::one() } else { tp / (tp + fp) });
            x0 = x;
        }
        p = p + F::one();
        if l { tp = tp + F::one(); }
        else { fp = fp + F::one(); }
    }
    recall.push(tp / ln);
    precision.push(tp / p);

    Some((precision, recall))
}

/// Computes the area under a PR curve of a given classifier.
///
/// `data` is a free-form `IntoIterator` object and `convert_fn` is a closure that converts each
/// data-point into a pair `(ground_truth, prediction).`
///
/// Returns `None` if one of the classes is not present or any values are non-finite.
/// Otherwise, returns `Some(area_under_curve)`.
pub fn pr_auc<T, X, F, I>(data: I, convert_fn: F) -> Option<X> where
        I: IntoIterator<Item=T>,
        F: Fn(T) -> (bool, X),
        X: Float {
    pr_auc_mut(&mut convert(data, convert_fn))
}

/// Computes the area under a PR curve of a given classifier sorting `pairs` in-place.
///
/// Returns `None` if one of the classes is not present or any values are non-finite.
/// Otherwise, returns `Some(area_under_curve)`.
pub fn pr_auc_mut<F: Float>(pairs: &mut [(bool, F)]) -> Option<F> {
    pr_mut(pairs).map(|curve| {
        trapezoidal(&curve.1, &curve.0)
    })
}

/// Computes the area under a ROC curve of a given classifier.
///
/// `data` is a free-form `IntoIterator` object and `convert_fn` is a closure that converts each
/// data-point into a pair `(ground_truth, prediction).`
///
/// Returns `None` if one of the classes is not present or any values are non-finite.
/// Otherwise, returns `Some(area_under_curve)`.
pub fn roc_auc<T, X, F, I>(data: I, convert_fn: F) -> Option<X> where
        I: IntoIterator<Item=T>,
        F: Fn(T) -> (bool, X),
        X: Float {
    roc_auc_mut(&mut convert(data, convert_fn))
}

/// Computes the area under a ROC curve of a given classifier sorting `pairs` in-place.
///
/// Returns `None` if one of the classes is not present or any values are non-finite.
/// Otherwise, returns `Some(area_under_curve)`.
pub fn roc_auc_mut<F: Float>(pairs: &mut [(bool, F)]) -> Option<F> {
    roc_mut(pairs).map(|curve| trapezoidal(&curve.0, &curve.1))
}
