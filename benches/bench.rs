#![feature(test)]

extern crate classifier_measures;
extern crate test;
extern crate rand;

use rand::Rng;
use test::Bencher;
use test::black_box as bb;

const NITEM: usize = 10_000;

macro_rules! bench {
    ($name:ident, $f:ident, $nitem:expr, $balance:expr) => (
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let mut data = (0..$nitem).map(|_| (rng.gen::<f64>() <= $balance, rng.gen())).collect::<Vec<(bool, f64)>>();
            b.iter(|| {
                bb(classifier_measures::$f(&mut data));
            });
        }
    );
    ($f:ident) => (bench!($f, $f, NITEM, 0.5););
}

bench!(pr_mut);
bench!(pr_auc_mut);
bench!(roc_mut);
bench!(roc_auc_mut);
bench!(pr_auc_mut_sparse);
bench!(pr_mut_sparse);

bench!(pr_auc_mut_sparse_imba, pr_auc_mut_sparse, 1_000_000, 0.000_03);
bench!(pr_auc_mut_imba, pr_auc_mut, 1_000_000, 0.000_03);
