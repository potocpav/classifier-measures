 #![feature(sort_unstable)]

// extern crate quickersort;
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

fn check_roc_auc_inputs<F: Float>(pairs: &[(bool, F)]) -> Result<(), &'static str> {
    if pairs.len() < 1 {
        return Err("Input is empty.");
    }

    if pairs.iter().find(|x| x.0 == true).is_none() || pairs.iter().find(|x| x.0 == false).is_none() {
        return Err("Both classes must be present.");
    }

    Ok(())
}

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

// sorts pairs in-place!
fn roc_unnormalized_mut<F: Float>(pairs: &mut [(bool, F)]) -> Option<(Vec<F>, Vec<F>)> {
    pairs.sort_unstable_by(&|x: &(_, F), y: &(_, F)|
        match y.1.partial_cmp(&x.1) {
            Some(ord) => ord,
            None      => panic!("A non-finite score is not allowed.")
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

    Some((fps, tps))
}

pub fn roc<T, X, F, I>(data: I, convert_fn: F) -> Option<(Vec<X>, Vec<X>)> where
        I: IntoIterator<Item=T>,
        F: Fn(T) -> (bool, X),
        X: Float {
    roc_mut(&mut convert(data, convert_fn))
}

pub fn roc_mut<F: Float>(pairs: &mut [(bool, F)]) -> Option<(Vec<F>, Vec<F>)> {
    if let Some((mut fps, mut tps)) = roc_unnormalized_mut(pairs) {
        let (tp_inv, fp_inv) = (F::one() / tps[tps.len() - 1], F::one() / fps[fps.len() - 1]);
        for mut x in tps.iter_mut() {
            *x = *x * tp_inv;
        }

        for mut y in fps.iter_mut() {
            *y = *y * fp_inv;
        }
        Some((fps, tps))
    } else {
        None
    }
}

pub fn pr<T, X, F, I>(data: I, convert_fn: F) -> Option<(Vec<X>, Vec<X>)> where
        I: IntoIterator<Item=T>,
        F: Fn(T) -> (bool, X),
        X: Float {
    pr_mut(&mut convert(data, convert_fn))
}

// sorts pairs in-place!
pub fn pr_mut<F: Float>(pairs: &mut [(bool, F)]) -> Option<(Vec<F>, Vec<F>)> {
    pairs.sort_unstable_by(&|x: &(_, F), y: &(_, F)|
        match y.1.partial_cmp(&x.1) {
            Some(ord) => ord,
            None      => unreachable!()
        });

    let mut x0 = F::nan();
    let (mut tp, mut p) = (F::zero(), F::zero());
    let (mut recall, mut precision) = (vec![], vec![]);

    // number of labels
    let ln = pairs.iter().fold(0, |a,b| a + if b.0 { 1 } else { 0 });
    let ln = NumCast::from(ln).unwrap();
    if ln == F::zero() {
        return None; // There is no positive sample
    }

    for &(l, x) in pairs.iter() {
        if x != x0 {
            recall.push(tp / ln);
            precision.push(if p == F::zero() { F::one() } else { tp / p });
            x0 = x;
        }
        p = p + F::one();
        if l { tp = tp + F::one(); }
    }
    recall.push(tp / ln);
    precision.push(tp / p);

    Some((precision, recall))
}

pub fn pr_auc<T, X, F, I>(data: I, convert_fn: F) -> Option<X> where
        I: IntoIterator<Item=T>,
        F: Fn(T) -> (bool, X),
        X: Float {
    pr_auc_mut(&mut convert(data, convert_fn))
}

pub fn pr_auc_mut<F: Float>(pairs: &mut [(bool, F)]) -> Option<F> {
    pr_mut(pairs).map(|curve| {
        trapezoidal(&curve.1, &curve.0)
    })
}

pub fn roc_auc<T, X, F, I>(data: I, convert_fn: F) -> Option<X> where
        I: IntoIterator<Item=T>,
        F: Fn(T) -> (bool, X),
        X: Float {
    roc_auc_mut(&mut convert(data, convert_fn))
}

pub fn roc_auc_mut<F: Float>(pairs: &mut [(bool, F)]) -> Option<F> {
    // check the input
    if let Err(_) = check_roc_auc_inputs(pairs) {
        return None;
    }
    roc_unnormalized_mut(pairs).map(|curve| {
        trapezoidal(&curve.0, &curve.1) / curve.0[curve.0.len()-1] / curve.1[curve.1.len()-1]
    })
}
